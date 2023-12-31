from src.agents.base_agent import BaseAgent


class DRQNAgent(BaseAgent):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.tau = float(config.train.tau)

    def optimize_model(self, experiences):
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

