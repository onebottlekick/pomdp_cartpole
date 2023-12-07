import argparse

import numpy as np

from utils.agent_utils import get_agent
from utils.env_utils import get_make_env_fn
from utils.logging_utils import mkExpDir
from utils.train_utils import save_results, save_weights, save_states, save_eval_states


def main(algorithm, config, logger):
    exp_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in config.train.seeds:
        logger.info(f'seed {seed}')
        agent = get_agent(algorithm)(config, logger)

        make_env_fn, make_env_kargs = get_make_env_fn(version=config.env.version, mdp=config.env.mdp, seed=seed, render=config.env.render)
        result, final_eval_score, training_time, wallclock_time, states, eval_states = agent.train(
            make_env_fn, make_env_kargs, seed, config.agent.gamma, config.agent.max_minutes, config.agent.max_episodes, config.agent.goal_mean_100_reward)
        save_results(np.array(result), config, seed=seed)
        save_states(states, config, seed=seed)
        save_eval_states(eval_states, config, seed=seed)
        exp_results.append(result)
        save_weights(agent.online_model, config, seed=seed)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
            
    save_weights(best_agent.online_model, config, best=True)
    save_results(np.array(exp_results), config, total=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='yaml file path')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    config, logger = mkExpDir(args.config, reset=args.reset)
    
    main(config.agent.type, config, logger)
