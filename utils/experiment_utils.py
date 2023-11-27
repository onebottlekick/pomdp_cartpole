import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.env_utils import get_make_env_fn


def load_results(experiment_name, seed=None):
    if seed is not None:
        path = os.path.join('experiments', experiment_name, 'results', experiment_name+'_seed'+str(seed)+'.npy')
    else:
        path = os.path.join('experiments', experiment_name, 'results', experiment_name+'_total.npy')
    
    results = np.load(path)
    
    if seed is not None:
        return results.T
    
    max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
    min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
    mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
    x = np.arange(np.max((len(mean_s), len(mean_s))))
    
    result_dict = {
        'total_steps':{
            'max' : max_t,
            'min' : min_t,
            'mean' : mean_t
        },
        'train_rewards':{
            'max' : max_r,
            'min' : min_r,
            'mean' : mean_r
        },
        'eval_scores':{
            'max' : max_s,
            'min' : min_s,
            'mean' : mean_s
        },
        'training_time':{
            'max' : max_sec,
            'min' : min_sec,
            'mean' : mean_sec
        },
        'wallclock_elapsed':{
            'max' : max_rt,
            'min' : min_rt,
            'mean' : mean_rt
        },
        'x': x
    }
    
    return result_dict


def plot_results(experiment_name, kind, idx=None, seeds=None, legend_loc='lower right', color='b', title=None, save=False, save_name=None):
    plt.style.use('seaborn-darkgrid')
    
    if seeds is not None:
        for seed in seeds:
            results = load_results(experiment_name, seed=seed)
            results = {
                      'total_steps': results[0],
                      'train_rewards': results[1],
                      'eval_scores': results[2],
                      'training_time': results[3],
                      'wallclock_elapsed': results[4]
                      }[kind]
            idxes = range(idx if idx is not None else len(results))
            plt.plot(results[idxes], linewidth=1.0, label=f'seed: {seed}')
        plt.legend(loc=legend_loc)
    
    else:
        results = load_results(experiment_name)

        max_, min_, mean_, x = results[kind]['max'], results[kind]['min'], results[kind]['mean'], results['x']

        plt.xlabel('Episodes')
        plt.plot(max_, color, linewidth=1.0)
        plt.plot(min_, color, linewidth=1.0)
        plt.plot(mean_, color+'--', linewidth=2.0)
        plt.fill_between(x, min_, max_, facecolor=color, alpha=0.3)
    
    if title is not None:
        plt.title(title)
    
    if save:
        save_dir = os.path.join('experiments', experiment_name, 'figures')
        os.makedirs(save_dir, exist_ok=True)
        if save_name is not None:
            save_name = os.path.join(save_dir, save_name + '.png')
        else:
            save_name = os.path.join(save_dir, experiment_name + '_' + kind + '.png')
        plt.savefig(save_name)
        plt.close()
        
    else:
        plt.show()
        plt.close()


def plot_states(config_path, ckpt_path=None, n_episodes=100, seed=None, render=False):
    import os
    
    from .agent_utils import get_agent
    from .config_utils import load_config
    
    config = load_config(config_path)
    
    agent = get_agent(config.agent.type)(config, None)
    
    env_fn, env_kwargs = get_make_env_fn(version=config.env.version, mdp=config.env.mdp, seed=seed, render=render)
    env = env_fn(**env_kwargs)
    
    model = agent.value_model_fn(env.n_observations, env.n_actions)
    network_ckpt = os.path.join('experiments', config.experiment.name, 'weights', f'{config.experiment.name}_best.pth') if ckpt_path is None else ckpt_path
    ckpt = torch.load(network_ckpt, map_location=model.device)
    model.load_state_dict(ckpt)
    model.eval()
    
    eval_strategy = agent.evaluation_strategy_fn()
    
    states = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        hidden_state = None
        cell_state = None
        while not done:
            states.append(state)
            if config.network.type in ['transformer', 'mtq']:
                action, hidden_state = eval_strategy.select_action(model, state, hidden_state)
            elif config.network.type in ['fcq', 'dueling_fcq']:
                action = eval_strategy.select_action(model, state)
            elif config.network.type in ['lstm']:
                action, hidden_state, cell_state = eval_strategy.select_action(model, state, hidden_state, cell_state)
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