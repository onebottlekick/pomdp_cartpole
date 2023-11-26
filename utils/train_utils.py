import os

import numpy as np
import torch
import torch.optim as optim


optimizer_dict = {
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
    'sgd': optim.SGD
}


def save_results(results, config, seed='', total=False):
    root = os.path.join('experiments', config.experiment.name, 'results')
    if total:
        path = os.path.join(root, f'{config.experiment.name}_total.npy')
    else:
        path = os.path.join(root, f'{config.experiment.name}_seed{seed}.npy')
    np.save(path, results)
    
    
def save_weights(model, config, seed='', best=False):
    root = os.path.join('experiments', config.experiment.name, 'weights')
    if best:
        path = os.path.join(root, f'{config.experiment.name}_best.pth')
    else:
        path = os.path.join(root, f'{config.experiment.name}_seed{seed}.pth')
    torch.save(model.state_dict(), path)