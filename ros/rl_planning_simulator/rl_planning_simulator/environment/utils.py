import numpy as np


def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = np.random.randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = np.random.rand()*(clip_high - clip_low) + clip_low
    return rand_num