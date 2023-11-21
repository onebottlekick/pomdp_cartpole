import os

import numpy as np


def load_results(algorithm_name, root='results'):
    path = os.path.join(root, algorithm_name+'.npy')
    results = np.load(path)
    max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
    min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
    mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
    x = np.arange(np.max((len(mean_s), len(mean_s))))
    result_dict = {
        't':{
            'max' : max_t,
            'min' : min_t,
            'mean' : mean_t
        },
        'r':{
            'max' : max_r,
            'min' : min_r,
            'mean' : mean_r
        },
        's':{
            'max' : max_s,
            'min' : min_s,
            'mean' : mean_s
        },
        'sec':{
            'max' : max_sec,
            'min' : min_sec,
            'mean' : mean_sec
        },
        'rt':{
            'max' : max_rt,
            'min' : min_rt,
            'mean' : mean_rt
        },
        'x': x
    }
    
    return result_dict