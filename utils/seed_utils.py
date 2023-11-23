import os
import random

import numpy as np
import torch


def seed_everything(seed, env):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)