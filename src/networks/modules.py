import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

DEFAULT_DIM_HEAD = 64
MIN_DIM_HEAD = 32


def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def exists(val):
    return val is not None


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

        
def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)
    
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset = 0):
        seq = torch.arange(max_seq_len, device = device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)


class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.g

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        null_kv = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * scale

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class BlockRecurrentAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_state,
        dim_head = DEFAULT_DIM_HEAD,
        state_len = 512,
        heads = 8,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        
        self.input_self_attn = Attention(dim, heads = heads, causal = True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads = heads, causal = False, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)

    def forward(
        self,
        x,
        state = None,
        mask = None,
        state_mask = None
    ):
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state), device=device)
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device = device)
        state_pos_emb = self.rotary_pos_emb(self.state_len, device = device)
        input_attn = self.input_self_attn(x, mask = mask, pos_emb = self_attn_pos_emb)
        state_attn = self.state_self_attn(state, mask = state_mask, pos_emb = state_pos_emb)

        input_as_q_cross_attn = self.input_state_cross_attn(x, context = state, mask = mask)
        state_as_q_cross_attn = self.state_input_cross_attn(state, context = x, mask = state_mask)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state
    
    
class BlockRecurrentTransformer(nn.Module):
    def __init__(self, dim, num_layers, memory_len, num_heads=8):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(BlockRecurrentAttention(dim, dim, state_len=memory_len, heads=num_heads))
    
    def forward(self, x, hidden_state=None):
        for attn in self.layers:
            x, hidden_state = attn(x, hidden_state)
        return x, hidden_state