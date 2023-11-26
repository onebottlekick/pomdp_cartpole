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
