import random
from collections import deque


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
            idx = random.randint(0, len(self.memory) - batch_size)
            e = list(self.memory)[idx:idx+batch_size]
            return e
        
    def available(self):
        return len(self.memory) > self.batch_size
    
    def __len__(self):
        return len(self.memory)