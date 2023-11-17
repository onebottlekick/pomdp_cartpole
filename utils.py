import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml

from env import CartPole


optimizer_dict = {
    'adam': optim.Adam,
    'rms_prop': optim.RMSprop,
    'sgd': optim.SGD
}


BEEP = lambda: os.system("printf '\a'")


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    
    
def save_results(results, algorithm_name, root='results'):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, algorithm_name)
    np.save(path, results)
    
    
def get_make_env_fn(**kargs):
    def make_env_fn(version, mdp, seed, render):
        env = CartPole(version=version, mdp=mdp, seed=seed, render=render)
        return env
    return make_env_fn, kargs


def load_results(algorithm_name, root='results'):
    path = os.path.join(root, algorithm_name+'.npy')
    results = np.load(path)
    max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
    min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
    mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
    x = np.arange(np.max((len(mean_s), len(mean_s))))
    result_dict = {
        't':{
            'max' : max_t,
            'min' : min_t,
            'mean' : mean_t
        },
        'r':{
            'max' : max_r,
            'min' : min_r,
            'mean' : mean_r
        },
        's':{
            'max' : max_s,
            'min' : min_s,
            'mean' : mean_s
        },
        'sec':{
            'max' : max_sec,
            'min' : min_sec,
            'mean' : mean_sec
        },
        'rt':{
            'max' : max_rt,
            'min' : min_rt,
            'mean' : mean_rt
        },
        'x': x
    }
    
    return result_dict


def plot_graph(algorithm_name, kind, color='b', title=None, save=False, save_name=None, root='results'):
    plt.style.use('seaborn-darkgrid')
    result_dict = load_results(algorithm_name, root)
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
        

def plot_states(network_ckpt, version='v0', n_episodes=100):
    from strategy import GreedyStrategy
    from network import Q_net
    
    config = network_ckpt.split('/')[-1].split('_')
    mdp = config[0]
    seq_len = int(config[2].strip('l[').strip(']'))
    dim = int(config[3].strip('d[').strip(']'))
    num_layers = int(config[4].strip('n[').strip('].pth'))
    n_observations = 4 if mdp == 'FOMDP' else 2
    n_actions = 2
    
    env_fn, env_kargs = get_make_env_fn(version=version, mdp=mdp, seed=None, render=False)
    env = env_fn(**env_kargs)
    
    model = Q_net(seq_len, dim, num_layers, n_observations, n_actions)
    ckpt = torch.load(network_ckpt, map_location=model.device)
    model.load_state_dict(ckpt)
    model.eval()
    
    eval_strategy = GreedyStrategy()
    
    states = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            states.append(state)
            action = eval_strategy.select_action(model, state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    env.close()
    del env
    
    if mdp == 'POMDP':
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