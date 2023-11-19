import numpy as np
import torch


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state, hidden_state):
        with torch.no_grad():
            # q_values = model(state).cpu().detach().data.numpy().squeeze()
            q_values, hidden_state = model(state, hidden_state)
            q_values = q_values.cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values), hidden_state
        
        
class EGreedyStrategy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state, hidden_state=None):
        self.exploratory_action_taken = False
        with torch.no_grad():
            # q_values = model(state).cpu().detach().data.numpy().squeeze()
            q_values, hidden_state = model(state, hidden_state)
            q_values = q_values.cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)
        return action, hidden_state
    
class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state, hidden_state=None):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values, hidden_state = model(state, hidden_state)
            q_values = q_values.cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action, hidden_state