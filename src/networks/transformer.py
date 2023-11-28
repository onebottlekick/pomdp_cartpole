import numpy as np
import torch
import torch.nn as nn

from .modules import BlockRecurrentTransformer


class MemoryTransformerQ(nn.Module):
    def __init__(self, memory_len, dim, num_layers, num_heads=8, n_observations=4, n_actions=2):
        super().__init__()
        self.dim = dim
        
        self.embed = nn.Linear(n_observations, dim)
        self.transformer = BlockRecurrentTransformer(dim, num_layers, memory_len=memory_len, num_heads=num_heads)
        self.V = nn.Linear(dim, 1)
        self.A = nn.Linear(dim, n_actions)
                
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
        x = x.view(1, -1, self.dim)
        x, hidden_state = self.transformer(x, hidden_state)
        x = x.view(-1, self.dim)
        advantage = self.A(x)
        value = self.V(x)
        Q = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return Q, hidden_state
    
    def load(self, experiences):
        states = np.vstack(experiences.state)
        actions = np.vstack(experiences.action)
        rewards = np.vstack(experiences.reward)
        new_states = np.vstack(experiences.next_state)
        is_terminals = np.vstack(experiences.done)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals