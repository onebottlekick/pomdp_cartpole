from src.agents.dqn import DQNAgent
from src.agents.ddqn import DDQNAgent
from src.agents.dueling_ddqn import DuelingDDQNAgent
from src.agents.drqn import DRQNAgent
from src.agents.transformer_dueling_ddqn import TransformerDuelingDDQNAgent
from src.agents.mtqn import MTQNAgent


def get_agent(algorithm):
    agent_dict = {
        'dqn': DQNAgent,
        'ddqn': DDQNAgent,
        'dueling_ddqn': DuelingDDQNAgent,
        'drqn': DRQNAgent,
        'transformer_dueling_ddqn': TransformerDuelingDDQNAgent,
        'mtqn': MTQNAgent,
    }
    algorithm = algorithm.rstrip('_pomdp')
    return agent_dict[algorithm]