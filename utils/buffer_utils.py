from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def make_epi_seq(episode, device):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for i in range(len(episode)):
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