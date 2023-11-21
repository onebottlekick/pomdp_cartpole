import os

import numpy as np
import torch
import torch.optim as optim

from src.agents.ddqn import DDQNAgent
from src.agents.dqn import DQNAgent
from src.agents.dueling_ddqn import DuelingDDQNAgent
from src.agents.transformer_dueling_ddqn import TransformerDuelingDDQNAgent
from src.buffer.episode_buffer import EpisodeBuffer
from src.buffer.replay_buffer import ReplayBuffer
from src.networks.linear import FCQ, FCDuelingQ
from src.networks.transformer import TransformerDuelingQ
from utils.config_utils import (ddqn_config, ddqn_pomdp_config, dqn_config,
                                dqn_pomdp_config, dueling_ddqn_config,
                                dueling_ddqn_pomdp_config,
                                transformer_dueling_ddqn_config,
                                transformer_dueling_ddqn_pomdp_config)
from utils.strategy import EGreedyExpStrategy, GreedyStrategy

network_dict = {
    'dqn': lambda nS, nA: FCQ(dim=dqn_config.network.in_dim, num_layers=dqn_config.network.num_layers, n_observations=nS, n_actions=nA),
    'ddqn': lambda nS, nA: FCQ(dim=ddqn_config.network.in_dim, num_layers=ddqn_config.network.num_layers, n_observations=nS, n_actions=nA),
    'dueling_ddqn': lambda nS, nA: FCDuelingQ(dim=dueling_ddqn_config.network.in_dim, num_layers=dueling_ddqn_config.network.num_layers, n_observations=nS, n_actions=nA),
    'transformer_dueling_ddqn': lambda nS, nA: TransformerDuelingQ(memory_len=transformer_dueling_ddqn_config.network.memory_len, dim=transformer_dueling_ddqn_config.network.in_dim, num_layers=transformer_dueling_ddqn_config.network.num_layers, n_observations=nS, n_actions=nA)
}


optimizer_dict = {
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
    'sgd': optim.SGD
}


agent_dict = {
    'dqn': DQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=dqn_config.train.batch_size),
        value_model_fn=network_dict[dqn_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[dqn_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(dqn_config.train.learning_rate),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=dqn_config.strategy.init_epsilon, min_epsilon=dqn_config.strategy.min_epsilon, decay_steps=dqn_config.strategy.decay_steps, net_type=dqn_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=dqn_config.network.net_type),
        n_warmup_batches=dqn_config.train.n_warmup_batches,
        update_target_every_steps=dqn_config.train.update_target_every_steps),
    
    'dqn_pomdp': DQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=dqn_pomdp_config.train.batch_size),
        value_model_fn=network_dict[dqn_pomdp_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[dqn_pomdp_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(dqn_pomdp_config.train.learning_rate),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=dqn_pomdp_config.strategy.init_epsilon,
                                                        min_epsilon=dqn_pomdp_config.strategy.min_epsilon,
                                                        decay_steps=dqn_pomdp_config.strategy.decay_steps,
                                                        net_type=dqn_pomdp_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=dqn_pomdp_config.network.net_type),
        n_warmup_batches=dqn_pomdp_config.train.n_warmup_batches,
        update_target_every_steps=dqn_pomdp_config.train.update_target_every_steps),
    
    'ddqn': DDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=ddqn_config.train.batch_size),
        value_model_fn=network_dict[ddqn_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[ddqn_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(ddqn_config.train.learning_rate),
        max_gradient_norm=float(ddqn_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=ddqn_config.strategy.init_epsilon,
                                                        min_epsilon=ddqn_config.strategy.min_epsilon,
                                                        decay_steps=ddqn_config.strategy.decay_steps,
                                                        net_type=ddqn_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=ddqn_config.network.net_type),
        n_warmup_batches=ddqn_config.train.n_warmup_batches,
        update_target_every_steps=ddqn_config.train.update_target_every_steps),
    
    'ddqn_pomdp': DDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=ddqn_pomdp_config.train.batch_size),
        value_model_fn=network_dict[ddqn_pomdp_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[ddqn_pomdp_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(ddqn_pomdp_config.train.learning_rate),
        max_gradient_norm=float(ddqn_pomdp_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=ddqn_pomdp_config.strategy.init_epsilon,
                                                        min_epsilon=ddqn_pomdp_config.strategy.min_epsilon,
                                                        decay_steps=ddqn_pomdp_config.strategy.decay_steps,
                                                        net_type=ddqn_pomdp_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=ddqn_pomdp_config.network.net_type),
        n_warmup_batches=ddqn_pomdp_config.train.n_warmup_batches,
        update_target_every_steps=ddqn_pomdp_config.train.update_target_every_steps),
    
    'dueling_ddqn': DuelingDDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=dueling_ddqn_config.train.batch_size),
        value_model_fn=network_dict[dueling_ddqn_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[dueling_ddqn_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(dueling_ddqn_config.train.learning_rate),
        max_gradient_norm=float(dueling_ddqn_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=dueling_ddqn_config.strategy.init_epsilon,
                                                        min_epsilon=dueling_ddqn_config.strategy.min_epsilon,
                                                        decay_steps=dueling_ddqn_config.strategy.decay_steps,
                                                        net_type=dueling_ddqn_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=dueling_ddqn_config.network.net_type),
        n_warmup_batches=dueling_ddqn_config.train.n_warmup_batches,
        update_target_every_steps=dueling_ddqn_config.train.update_target_every_steps,
        tau=dueling_ddqn_config.train.tau),
    
    'dueling_ddqn_pomdp': DuelingDDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=dueling_ddqn_pomdp_config.train.batch_size),
        value_model_fn=network_dict[dueling_ddqn_pomdp_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[dueling_ddqn_pomdp_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(dueling_ddqn_pomdp_config.train.learning_rate),
        max_gradient_norm=float(dueling_ddqn_pomdp_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=dueling_ddqn_pomdp_config.strategy.init_epsilon,
                                                        min_epsilon=dueling_ddqn_pomdp_config.strategy.min_epsilon,
                                                        decay_steps=dueling_ddqn_pomdp_config.strategy.decay_steps,
                                                        net_type=dueling_ddqn_pomdp_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=dueling_ddqn_pomdp_config.network.net_type),
        n_warmup_batches=dueling_ddqn_pomdp_config.train.n_warmup_batches,
        update_target_every_steps=dueling_ddqn_pomdp_config.train.update_target_every_steps,
        tau=dueling_ddqn_pomdp_config.train.tau),
    
    'transformer_dueling_ddqn': TransformerDuelingDDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=transformer_dueling_ddqn_config.train.seq_len),
        episode_buffer_fn=lambda: EpisodeBuffer(batch_size=transformer_dueling_ddqn_config.train.batch_size, random_sample=transformer_dueling_ddqn_config.train.random_sample),
        value_model_fn=network_dict[transformer_dueling_ddqn_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[transformer_dueling_ddqn_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(transformer_dueling_ddqn_config.train.learning_rate),
        max_gradient_norm=float(transformer_dueling_ddqn_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=transformer_dueling_ddqn_config.strategy.init_epsilon,
                                                        min_epsilon=transformer_dueling_ddqn_config.strategy.min_epsilon,
                                                        decay_steps=transformer_dueling_ddqn_config.strategy.decay_steps,
                                                        net_type=transformer_dueling_ddqn_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=transformer_dueling_ddqn_config.network.net_type),
        n_warmup_batches=transformer_dueling_ddqn_config.train.n_warmup_batches,
        update_target_every_steps=transformer_dueling_ddqn_config.train.update_target_every_steps,
        tau=transformer_dueling_ddqn_config.train.tau),
    
    'transformer_dueling_ddqn_pomdp': TransformerDuelingDDQNAgent(
        replay_buffer_fn=lambda: ReplayBuffer(seq_len=transformer_dueling_ddqn_pomdp_config.train.seq_len),
        episode_buffer_fn=lambda: EpisodeBuffer(batch_size=transformer_dueling_ddqn_pomdp_config.train.batch_size, random_sample=transformer_dueling_ddqn_pomdp_config.train.random_sample),
        value_model_fn=network_dict[transformer_dueling_ddqn_pomdp_config.network.model],
        value_optimizer_fn=lambda net, lr: optimizer_dict[transformer_dueling_ddqn_pomdp_config.train.optimizer](net.parameters(), lr=lr),
        value_optimizer_lr=float(transformer_dueling_ddqn_pomdp_config.train.learning_rate),
        max_gradient_norm=float(transformer_dueling_ddqn_pomdp_config.train.max_gradient_norm),
        training_strategy_fn=lambda: EGreedyExpStrategy(init_epsilon=transformer_dueling_ddqn_pomdp_config.strategy.init_epsilon,
                                                        min_epsilon=transformer_dueling_ddqn_pomdp_config.strategy.min_epsilon,
                                                        decay_steps=transformer_dueling_ddqn_pomdp_config.strategy.decay_steps,
                                                        net_type=transformer_dueling_ddqn_pomdp_config.network.net_type),
        evaluation_strategy_fn=lambda: GreedyStrategy(net_type=transformer_dueling_ddqn_pomdp_config.network.net_type),
        n_warmup_batches=transformer_dueling_ddqn_pomdp_config.train.n_warmup_batches,
        update_target_every_steps=transformer_dueling_ddqn_pomdp_config.train.update_target_every_steps,
        tau=transformer_dueling_ddqn_pomdp_config.train.tau)
}


def beep():
    os.system("printf '\a'")


def save_results(results, algorithm_name, mdp, root='results', seed='', total=False):
    root = os.path.join(root, mdp, algorithm_name)
    os.makedirs(root, exist_ok=True)
    if total:
        path = os.path.join(root, f'{algorithm_name}_total.npy')
    else:
        path = os.path.join(root, f'{algorithm_name}_seed{seed}.npy')
    np.save(path, results)
    
    
def save_weights(model, algorithm_name, mdp, root='weights', seed='', best=False):
    root = os.path.join(root, mdp, algorithm_name)
    os.makedirs(root, exist_ok=True)
    if best:
        path = os.path.join(root, f'{algorithm_name}_best.pth')
    else:
        path = os.path.join(root, f'{algorithm_name}_seed{seed}.pth')
    torch.save(model.state_dict(), path)