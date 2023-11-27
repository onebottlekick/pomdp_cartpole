import gc
import glob
import os
import tempfile
import time
from itertools import count

import numpy as np
import torch

from src.buffer.replay_buffer import ReplayBuffer
from utils.buffer_utils import Transition
from utils.seed_utils import seed_everything
from utils.strategy import EGreedyExpStrategy, GreedyStrategy
from utils.network_utils import get_network
from utils.train_utils import optimizer_dict


class BaseAgent:
    def __init__(self, config, logger):        
        self.replay_buffer_fn = lambda: ReplayBuffer(seq_len=config.train.batch_size)
        self.value_model_fn = get_network(config)
        self.value_optimizer_fn = lambda net, lr: optimizer_dict[config.train.optimizer](net.parameters(), lr=lr)
        self.value_optimizer_lr = float(config.train.learning_rate)
        self.training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=config.strategy.init_epsilon,
                                                       min_epsilon=config.strategy.min_epsilon,
                                                       decay_steps=config.strategy.decay_steps,
                                                       type=config.network.type)
        self.evaluation_strategy_fn = lambda: GreedyStrategy(type=config.network.type)
        self.n_warmup_batches = config.train.n_warmup_batches
        self.update_target_every_steps = config.train.update_target_every_steps
        self.tau = 1.0

        self.is_transformer = config.network.type in ['transformer', 'mtq']
        self.is_linear = config.network.type in ['fcq', 'dueling_fcq']
        self.is_lstm = config.network.type in ['lstm']
        self.__logger = logger

    def optimize_model(self):
        raise NotImplementedError

    def interaction_step(self, state, env, hidden_state=None, cell_state=None):
        if self.is_transformer:
            action, hidden_state = self.training_strategy.select_action(self.online_model, state, hidden_state)
        elif self.is_linear:
            action = self.training_strategy.select_action(self.online_model, state)
        elif self.is_lstm:
            action, hidden_state, cell_state = self.training_strategy.select_action(self.online_model, state, hidden_state, cell_state)
        new_state, reward, terminated, truncated, info = env.step(action)
        is_failure = terminated
        is_terminal = terminated or truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.push(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        
        if self.is_transformer:
            return new_state, is_terminal, hidden_state
        elif self.is_linear:
            return new_state, is_terminal
        elif self.is_lstm:
            return new_state, is_terminal, hidden_state, cell_state
    
    def update_network(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start = time.time()

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs)
        seed_everything(self.seed, env=env)
    
        nS, nA = env.n_observations, env.n_actions
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network(tau=1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn() 
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, _ = env.reset(seed=self.seed)
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            hidden_state, cell_state = None, None
            for step in count():
                if self.is_transformer:
                    state, is_terminal, hidden_state = self.interaction_step(state, env, hidden_state)
                elif self.is_linear:
                    state, is_terminal = self.interaction_step(state, env)
                elif self.is_lstm:
                    state, is_terminal, hidden_state, cell_state = self.interaction_step(state, env, hidden_state, cell_state)
                
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = Transition(*zip(*experiences))
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()
                
                if is_terminal:
                    gc.collect()
                    break
            
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.save_checkpoint(episode-1, self.online_model)
            
            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            cur_eval_score = self.evaluation_scores[-1]
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed
            
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f} '
            debug_message += 'cur_ev {:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score, cur_eval_score)
            self.__logger.info(debug_message)
            
            if training_is_over:
                if reached_max_minutes: self.__logger.info(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: self.__logger.info(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: self.__logger.info(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        self.__logger.info('Training complete.')
        self.__logger.info('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, _ = eval_env.reset()
            d = False
            rs.append(0)
            h, c = None, None
            for _ in count():
                if self.is_transformer:
                    a, h = self.evaluation_strategy.select_action(eval_policy_model, s, h)
                elif self.is_linear:
                    a = self.evaluation_strategy.select_action(eval_policy_model, s)
                elif self.is_lstm:
                    a, h, c = self.evaluation_strategy.select_action(eval_policy_model, s, h, c)
                s, r, terminated, truncated, _ = eval_env.step(a)
                d = terminated or truncated
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))