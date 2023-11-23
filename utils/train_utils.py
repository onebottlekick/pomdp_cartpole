import os

import numpy as np
import torch
import torch.optim as optim


optimizer_dict = {
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
    'sgd': optim.SGD
}


def beep():
    os.system("printf '\a'")


def save_results(results, algorithm_name, mdp, root='results', seed='', total=False):
    root = os.path.join(root, mdp, algorithm_name)
    os.makedirs(root, exist_ok=True)
    if total:
        path = os.path.join(root, f'{algorithm_name}_total.npy')
    else:
        path = os.path.join(root, f'{algorithm_name}_seed{seed}.npy')
    np.save(path, results)
    
    
def save_weights(model, algorithm_name, mdp, root='weights', seed='', best=False):
    root = os.path.join(root, mdp, algorithm_name)
    os.makedirs(root, exist_ok=True)
    if best:
        path = os.path.join(root, f'{algorithm_name}_best.pth')
    else:
        path = os.path.join(root, f'{algorithm_name}_seed{seed}.pth')
    torch.save(model.state_dict(), path)