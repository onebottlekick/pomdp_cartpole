import argparse

import yaml


def load_config_from_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def config_to_namespace(config):
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = config_to_namespace(value)
    return argparse.Namespace(**config)


def load_config(path):
    config = load_config_from_yaml(path)
    return config_to_namespace(config)