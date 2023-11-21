import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.env_utils import get_make_env_fn
from utils.experiment_utils import load_results


def plot_graph(algorithm_name, mdp, kind, color='b', title=None, save=False, save_name=None, root='results'):
    plt.style.use('seaborn-darkgrid')
    result_dict = load_results(algorithm_name, mdp, root)
    max_, min_, mean_, x = result_dict[kind]['max'], result_dict[kind]['min'], result_dict[kind]['mean'], result_dict['x']
    
    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes')
    plt.plot(max_, color, linewidth=1.0)
    plt.plot(min_, color, linewidth=1.0)
    plt.plot(mean_, color+'--', linewidth=2.0)
    plt.fill_between(x, min_, max_, facecolor=color, alpha=0.3)
    
    if save:
        save_dir = os.path.join('train', root, 'figure')
        os.makedirs(save_dir, exist_ok=True)
        if save_name is not None:
            save_name = os.path.join(save_dir, save_name + '.png')
        else:
            save_name = os.path.join(save_dir, algorithm_name + '_' + kind + '.png')
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_states(algorithm, n_episodes=100, render=False):
    import os
    
    from .train_utils import agent_dict
    from .config_utils import config_dict
    
    config = config_dict[algorithm]
    
    agent = agent_dict[config.network.model]
    
    env_fn, env_kwargs = get_make_env_fn(version=config.env.version, mdp=config.env.mdp, seed=None, render=render)
    env = env_fn(**env_kwargs)
    
    model = agent.value_model_fn(env.n_observations, env.n_actions)
    network_ckpt = os.path.join('weights', config.env.mdp, algorithm, f'{algorithm}_best.pth')
    ckpt = torch.load(network_ckpt, map_location=model.device)
    model.load_state_dict(ckpt)
    model.eval()
    
    eval_strategy = agent.evaluation_strategy_fn()
    
    states = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        hidden_state = None
        while not done:
            states.append(state)
            if config.network.net_type == 'transformer':
                action, hidden_state = eval_strategy.select_action(model, state, hidden_state)
            elif config.network.net_type == 'linear':
                action = eval_strategy.select_action(model, state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    env.close()
    del env
    
    if config.env.mdp == 'POMDP':
        x = np.array(states)[:, 0]
        a = np.array(states)[:, 1]

    else:
        x = np.array(states)[:, 0]
        a = np.array(states)[:, 2]
    
    plt.style.use('seaborn-darkgrid')
    plt.plot(x)
    plt.xlabel('Steps')
    plt.ylabel('Cart Position')
    plt.show()
    
    plt.plot(a)
    plt.xlabel('Steps')
    plt.ylabel('Pole Angle')
    plt.show()