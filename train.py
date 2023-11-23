import argparse

import numpy as np

from utils.config_utils import load_config
from utils.env_utils import get_make_env_fn
from utils.train_utils import beep, save_results, save_weights
from utils.agent_utils import get_agent


def main(algorithm, config):
    exp_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in config.train.seeds:
        agent = get_agent(algorithm)(config)

        make_env_fn, make_env_kargs = get_make_env_fn(version=config.env.version, mdp=config.env.mdp, seed=seed, render=config.env.render)
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed, config.agent.gamma, config.agent.max_minutes, config.agent.max_episodes, config.agent.goal_mean_100_reward)
        save_results(np.array(result), algorithm_name=config.network.model, mdp=config.env.mdp, seed=seed)
        exp_results.append(result)
        save_weights(agent.online_model, algorithm_name=config.network.model, mdp=config.env.mdp, seed=seed)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
            
    save_weights(best_agent.online_model, algorithm_name=config.network.model, mdp=config.env.mdp, best=True)
    beep()
    save_results(np.array(exp_results), algorithm_name=config.network.model, mdp=config.env.mdp, total=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str, help='Name of the algorithm to run')
    arg = parser.parse_args()
    
    config_path = f'configs/{arg.algorithm}.yml'
    config = load_config(config_path)
    print(config)
    
    main(arg.algorithm, config)
    
    