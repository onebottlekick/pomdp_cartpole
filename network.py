import torch
import torch.nn as nn

from module import BlockRecurrentTransformer


class Q_net(nn.Module):
    def __init__(self, seq_len, dim, num_layers, n_observations=4, n_actions=2):
        super().__init__()
        
        self.embed = nn.Linear(n_observations, dim)
        self.transformer = BlockRecurrentTransformer(dim, num_layers, seq_len)
        self.V = nn.Linear(dim, 1)
        self.A = nn.Linear(dim, n_actions)
        # self.q = nn.Linear(dim, n_actions)
        
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        self.device = device
        self.to(device)
    
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
        x, hidden_state = self.transformer(x)
        advantage = self.A(x)
        value = self.V(x)
        Q = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return Q