from src.networks.linear import FCQ, FCDuelingQ
from src.networks.lstm import LSTMQ
from src.networks.transformer import TransformerDuelingQ, MemoryTransformerQ


def get_network(config):
    network_dict = {
        'fcq': lambda nS, nA: FCQ(dim=config.network.in_dim,
                                    num_layers=config.network.num_layers,
                                    n_observations=nS,
                                    n_actions=nA),
        
        'dueling_fcq': lambda nS, nA: FCDuelingQ(dim=config.network.in_dim,
                                                    num_layers=config.network.num_layers,
                                                    n_observations=nS,
                                                    n_actions=nA),
        
        'transformer': lambda nS, nA: TransformerDuelingQ(memory_len=config.network.memory_len,
                                                            dim=config.network.in_dim,
                                                            num_layers=config.network.num_layers,
                                                            num_heads=config.network.num_heads,
                                                            n_observations=nS,
                                                            n_actions=nA),
        
        'lstm': lambda nS, nA: LSTMQ(dim=config.network.in_dim,
                                        num_layers=config.network.num_layers,
                                        n_observations=nS,
                                        n_actions=nA),
        
        'mtq': lambda nS, nA: MemoryTransformerQ(memory_len=config.network.memory_len,
                                        dim=config.network.in_dim,
                                        num_layers=config.network.num_layers,
                                        num_heads=config.network.num_heads,
                                        n_observations=nS,
                                        n_actions=nA),
    }
    
    return network_dict[config.network.type]