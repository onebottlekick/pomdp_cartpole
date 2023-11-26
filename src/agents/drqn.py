from itertools import count

import numpy as np

from src.agents.base_agent import BaseAgent


class DRQNAgent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def optimize_model(self, experiences, h=None, c=None):
        states, actions, rewards, next_states, is_terminals = experiences
        
        max_a_q_sp, _, _ = self.target_model(next_states)
        max_a_q_sp = max_a_q_sp.detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa, _, _ = self.online_model(states)
        q_sa = q_sa.gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env, h=None, c=None):
        action, h, c = self.training_strategy.select_action(self.online_model, state, h, c)
        new_state, reward, terminated, truncated, _ = env.step(action)
        is_failure = terminated
        is_terminal = terminated or truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.push(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal, h, c
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, _ = eval_env.reset(seed=self.seed)
            d = False
            rs.append(0)
            h, c = None, None
            for _ in count():
                a, h, c = self.evaluation_strategy.select_action(eval_policy_model, s, h, c)
                s, r, terminated, truncated, _ = eval_env.step(a)
                d = terminated or truncated
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)
