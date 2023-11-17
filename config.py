from utils import load_config


config = load_config('config.yml')


#########################Experiment Configuration########################
exp_name = config['experiment']['name']
model_dir = config['experiment']['model_dir']
model_name = config['experiment']['model_name']
result_dir = config['experiment']['result_dir']
result_name = config['experiment']['result_name']
#########################################################################


###########################Train Configuration###########################
seeds = config['train']['seeds']
batch_size = config['train']['batch_size']
value_optimizer_lr = config['train']['learning_rate']
optimizer = config['train']['optimizer']
max_gradient_norm = float(config['train']['max_gradient_norm'])
n_warmup_batches = config['train']['n_warmup_batches']
update_target_every_steps = config['train']['update_target_every_steps']
tau = config['train']['tau']
#########################################################################


##########################Network Configuration##########################
seq_len = config['network']['seq_len']
hidden_size = config['network']['in_dim']
num_layers = config['network']['num_layers']
#########################################################################


########################Environment Configuration########################
version = config['env']['version']
goal_mean_100_reward = config['env']['goal_mean_100_reward']
mdp = config['env']['mdp']
render = config['env']['render']
#########################################################################


###########################Agent Configuration###########################
gamma = config['agent']['gamma']
max_minutes = config['agent']['max_minutes']
max_episodes = config['agent']['max_episodes']
#########################################################################


#########################Strategy Configuration##########################
init_epsilon = config['strategy']['init_epsilon']
min_epsilon = config['strategy']['min_epsilon']
decay_steps = config['strategy']['decay_steps']
#########################################################################