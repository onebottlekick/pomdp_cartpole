import torch
import torch.nn as nn

from .modules import BlockRecurrentTransformer


class TransformerDuelingQ(nn.Module):
    def __init__(self, memory_len, dim, num_layers, n_observations=4, n_actions=2):
        super().__init__()
        
        self.embed = nn.Linear(n_observations, dim)
        self.transformer = BlockRecurrentTransformer(dim, num_layers, memory_len=memory_len)
        self.bottle_neck = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Linear(512, 128)
        )
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)
                
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        self.device = device
        
        self.apply(self._init_weights)
        self.to(device)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
    
    def forward(self, x, hidden_state=None):
        x = self._format(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.embed(x)
        x, hidden_state = self.transformer(x, hidden_state)
        x = self.bottle_neck(x)
        advantage = self.A(x)
        value = self.V(x)
        Q = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return Q, hidden_state