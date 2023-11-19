import os

import numpy as np
import torch

from agent import DuelingDDQN
from buffer import EpisodeBuffer, ReplayBuffer
from config import *
from network import Q_net
from strategy import EGreedyExpStrategy, GreedyStrategy
from utils import (BEEP, get_make_env_fn, optimizer_dict,
                   save_results)

exp_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in seeds:
    value_model_fn = lambda nS, nA: Q_net(memory_len=memory_len, dim=hidden_size, num_layers=num_layers, n_observations=nS, n_actions=nA)
    value_optimizer_fn = lambda net, lr: optimizer_dict[optimizer](net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    max_gradient_norm = float('inf')

    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=init_epsilon,  
                                                      min_epsilon=min_epsilon, 
                                                      decay_steps=decay_steps)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(seq_len=seq_len, random_sample=random_sample)
    episode_buffer_fn = lambda: EpisodeBuffer(batch_size=batch_size, random_sample=random_sample)

    agent = DuelingDDQN(replay_buffer_fn,
                episode_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                max_gradient_norm,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_target_every_steps,
                tau)

    make_env_fn, make_env_kargs = get_make_env_fn(version=version, mdp=mdp, seed=seed, render=render)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    exp_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
exp_results = np.array(exp_results)
os.makedirs(model_dir, exist_ok=True)
torch.save(best_agent.online_model.state_dict(), os.path.join(model_dir, f'{model_name}_b[{batch_size}]_l[{seq_len}]_d[{hidden_size}]_n[{num_layers}].pth'))
_ = BEEP()
save_results(exp_results, algorithm_name=f'{result_name}_b[{batch_size}]_l[{seq_len}]_d[{hidden_size}]_n[{num_layers}]', root=result_dir)