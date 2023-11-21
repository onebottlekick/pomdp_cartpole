import argparse

import yaml


def load_config_form_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def config_to_namespace(config):
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = config_to_namespace(value)
    return argparse.Namespace(**config)


def load_config(path):
    config = load_config_form_yaml(path)
    return config_to_namespace(config)

dqn_config = load_config(path='configs/dqn.yml')
dqn_pomdp_config = load_config(path='configs/dqn_pomdp.yml')

ddqn_config = load_config(path='configs/ddqn.yml')
ddqn_pomdp_config = load_config(path='configs/ddqn_pomdp.yml')

dueling_ddqn_config = load_config(path='configs/dueling_ddqn.yml')
dueling_ddqn_pomdp_config = load_config(path='configs/dueling_ddqn_pomdp.yml')

transformer_dueling_ddqn_config = load_config(path='configs/transformer_dueling_ddqn.yml')
transformer_dueling_ddqn_pomdp_config = load_config(path='configs/transformer_dueling_ddqn_pomdp.yml')


config_dict = {
    'dqn': dqn_config,
    'dqn_pomdp': dqn_pomdp_config,
    'ddqn': ddqn_config,
    'ddqn_pomdp': ddqn_pomdp_config,
    'dueling_ddqn': dueling_ddqn_config,
    'dueling_ddqn_pomdp': dueling_ddqn_pomdp_config,
    'transformer_dueling_ddqn': transformer_dueling_ddqn_config,
    'transformer_dueling_ddqn_pomdp': transformer_dueling_ddqn_pomdp_config
}

if __name__ == "__main__":
    print(type(transformer_dueling_ddqn_config.train.learning_rate))

