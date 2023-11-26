from itertools import count

import numpy as np
import torch

from src.agents.base_agent import BaseAgent


class MTQNAgent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.max_gradient_norm = float(config.train.max_gradient_norm)
        self.tau = config.train.tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        a_q_sp, _ = self.online_model(next_states)
        argmax_a_q_sp = a_q_sp.max(1)[1]
        q_sp, _ = self.target_model(next_states)
        q_sp = q_sp.detach()
        
        max_a_q_sp = q_sp[
            np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa, _ = self.online_model(states)
        q_sa = q_sa.gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.max_gradient_norm)
        self.value_optimizer.step()

    def interaction_step(self, state, env, hidden_state=None):
        action, hidden_state = self.training_strategy.select_action(self.online_model, state, hidden_state)
        new_state, reward, terminated, truncated, info = env.step(action)
        is_failure = terminated
        is_terminal = terminated or truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.push(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal, hidden_state
    
    def update_network(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, _ = eval_env.reset()
            d = False
            rs.append(0)
            hidden_state = None
            for _ in count():
                a, hidden_state = self.evaluation_strategy.select_action(eval_policy_model, s, hidden_state)
                s, r, terminated, truncated, _ = eval_env.step(a)
                d = terminated or truncated
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)
