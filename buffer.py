from collections import namedtuple, deque
import random

import torch
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def make_epi_seq(episode, batch_size, device):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for i in range(batch_size):
        t = Transition(*zip(*episode[i]))
        states.append(t.state)
        actions.append(t.action)
        rewards.append(t.reward)
        next_states.append(t.next_state)
        dones.append(t.done)
        
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    
    states = torch.tensor(states).float().to(device)
    actions = torch.tensor(actions).unsqueeze(-1).long().to(device)
    rewards = torch.tensor(rewards).unsqueeze(-1).float().to(device)
    next_states = torch.tensor(next_states).float().to(device)
    dones = torch.tensor(dones).unsqueeze(-1).float().to(device)
    
    return states, actions, rewards, next_states, dones


class ReplayBuffer:
    def __init__(self, capacity=10000, seq_len=64, random_sample=True):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = seq_len
        self.random_sample = random_sample
        
    def push(self, args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.random_sample:
            return random.sample(self.memory, self.batch_size)
        else:
            return self.memory[-batch_size:]
    
    def __len__(self):
        return len(self.memory)
    
    
class EpisodeBuffer:
    def __init__(self, capacity=10000, batch_size=32, random_sample=True):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.random_sample = random_sample
        
    def push(self, episode):
        self.memory.append(episode)
        
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.random_sample:
            return random.sample(self.memory, self.batch_size)
        else:
            return self.memory[-batch_size:]
        
    def available(self):
        return len(self.memory) > self.batch_size
    
    def __len__(self):
        return len(self.memory)