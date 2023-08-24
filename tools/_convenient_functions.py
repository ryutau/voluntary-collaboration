from scipy.special import comb
import numpy as np
import pandas as pd
import random
import math


def p_k(n, gamma, k):
    return comb(n, k) * (gamma ** k) * ((1 - gamma) ** (n - k))


def floor_float(x, n):
    if np.isnan(x):
        return x
    else:
        new_x = math.floor(x * 10 ** n) / (10 ** n)
        return new_x


def bootstrap_sampling(original_exp_data, random_seed):
    """
    Resample experimental data with the participant as a unit.

    Parameters
    ----------
    original_exp_data: pandas.DataFrame
        Original experimental data that contains participant identification code (pid) and all the necessary columns.
    random_seed: int
        Random seed to secure the reproducibility.

    Returns
    -------
    sampled_exp_data: pandas.DataFrame
        Resampled experimental data with its shape identical to the original_exp_data
    """
    random.seed(random_seed)
    unique_pids = original_exp_data["pid"].unique()
    sampled_pids = random.choices(unique_pids, k=len(unique_pids)) # sample with replacement
    sampled_exp_data = pd.concat([original_exp_data[lambda x: x["pid"] == pid] for pid in sampled_pids]).reset_index(drop=True)
    return sampled_exp_data
