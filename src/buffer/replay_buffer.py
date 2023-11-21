import random
from collections import deque

from utils.buffer_utils import Transition


class ReplayBuffer:
    def __init__(self, capacity=10000, seq_len=64, random_sample=True):
        self.memory = deque([], maxlen=capacity)
        self.seq_len = seq_len
        self.batch_size = seq_len
        self.random_sample = random_sample
        
        self.capacity = capacity
        
        self.pos = 0
        
    def push(self, args):
        self.memory.append(Transition(*args))
        
    def sample(self, seq_len=None):
        if seq_len == None:
            seq_len = self.seq_len
        
        if self.random_sample:
            return random.sample(self.memory, seq_len)

        else:
            idx = random.randint(0, len(self.memory) - seq_len)
            e = list(self.memory)[idx:idx+seq_len]
            return e             
    
    def __len__(self):
        return len(self.memory)