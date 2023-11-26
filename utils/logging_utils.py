import logging
import os
import shutil

import yaml

from utils.config_utils import load_config


class Logger:
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
    
    
def mkExpDir(config_path, reset=False):
    root = 'experiments'
    config = load_config(config_path)
    exp_root = os.path.join(root, config.experiment.name)
    
    if os.path.exists(exp_root):
        if reset:
            shutil.rmtree(exp_root)   
        
    result_root = os.path.join(exp_root, 'results')
    weights_root = os.path.join(exp_root, 'weights')
    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(weights_root, exist_ok=True)
    
    shutil.copyfile(config_path, os.path.join(exp_root, f'{config.experiment.name}.yml'))

    _logger = Logger(log_file_name=os.path.join(exp_root, config.experiment.name+'.log'),
        logger_name=config.experiment.name).get_log()

    return config, _logger