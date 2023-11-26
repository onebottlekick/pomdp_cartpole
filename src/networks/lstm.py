import numpy as np
import torch
import torch.nn as nn
    
    
class LSTMQ(nn.Module):
    def __init__(self, dim, num_layers, n_observations=4, n_actions=2):
        super().__init__()
        self.dim = dim
        
        self.activation = nn.ReLU()
        
        self.fc1 = nn.Linear(n_observations, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.out = nn.Linear(dim, n_actions)
        
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
            
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
    
    def forward(self, x, h=None, c=None):
        x = self._format(x)
        x = self.activation(self.fc1(x))
        x = x.view(1, -1, self.dim)
        if h is None:
            h = torch.zeros(1, 1, self.dim).to(self.device)
        if c is None:
            c = torch.zeros(1, 1, self.dim).to(self.device)
            
        x, (h, c) = self.lstm(x, (h, c))
        x = x.view(-1, self.dim)
        x = self.activation(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.activation(x)
        x = self.out(x)
        return x, h, c
    
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