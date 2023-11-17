import gym


class CartPole:
    def __init__(self, version='v0', mdp='FOMDP', seed=None, render=False):
        render_mode = 'human' if render else None
        self.env = gym.make('CartPole-'+version, render_mode=render_mode)
        
        assert mdp in ['FOMDP', 'POMDP']
        self.mdp = mdp
        
        self.observation_space = self.env.observation_space
        self.n_observations = 4 if mdp == 'FOMDP' else 2
        
        self.action_space = self.env.action_space
        self.n_actions = 2
        
        self._seed = seed
        
    def reset(self):
        state, info = self.env.reset(seed=self._seed)
        if self.mdp == 'POMDP':
            state = state[[0, 2]]
        return state, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.mdp == 'POMDP':
            state = state[[0, 2]]
        return state, reward, terminated, truncated, info
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()
        
    def seed(self, seed):
        self.seed = seed
        
        
if __name__ == '__main__':
    from itertools import count
    
    mdp = 'FOMDP'
    
    env = CartPole(mdp=mdp)
    
    for epi in range(100):
        for step in count():
            state, info = env.reset()
            env.render()
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            assert state.shape == (2,) if mdp == 'POMDP' else (4,)
            
            if done:
                break