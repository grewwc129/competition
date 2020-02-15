import pandas as pd
from .config import config
from .maths import *
import numpy as np

class_label = config.class_label  # possibly faster


def standardize(arr2d, minmax=True):
    if minmax:
        min_val, max_val = arr2d.min(axis=1), arr2d.max(axis=1)
        diff = max_val - min_val
        return (arr2d - min_val) / (diff+1e-8)

    mean, std = arr2d.mean(axis=1), arr2d.std(axis=1)
    return (arr2d - mean) / (std+1e-8)


def encode_name(name):
    return class_label[name]


def trim_df(arr2d, use_bin=True, remove_endpoints=True, flatten=True, remove_bad=True):
    """
    use_bin:
        if True, use "average_bin" function
    returns:
        train_x: (None, Dim, 1)
        train_y: (Dim, )
    """
    num_remove_begin = 100
    num_remove_end = 50
    # first remove id, id is the last column
    arr2d = arr2d.iloc[:, :-1]
    # because the data label is wrong
    arr2d = arr2d[arr2d.iloc[:, -1] != "answer"]
    # then split "answer" and other columns
    # "answer" is the last second column, the last one for now
    train_x = arr2d.iloc[:, :-1].astype(np.float).values
    # train_x = standardize(train_x, minmax=False).values

    train_y = arr2d.iloc[:, -1].apply(encode_name).astype(np.int8).values

    if remove_endpoints:
        train_x = train_x[:, num_remove_begin:-num_remove_end]

    if remove_bad:
        train_x, train_y = find_bad(train_x, train_y, return_bad=False)

    if flatten:
        train_x = remove_sharp(train_x, threshold=10)
        # train_x = change_zero_to_mean(train_x)

    if use_bin:
        # train_x = average_bin_faster(train_x, num_bins=config.num_bins)
        train_x = median_bin_faster(train_x, num_bins=config.num_bins)

    train_x = train_x.reshape(*train_x.shape, 1)
    return train_x, train_y


def trim_df_binary(arr2d, target_name, use_bin=True, remove_begin=True, remove_end=True):
    """
    use_bin:
        if True, use "average_bin" function
    returns:
        train_x: (None, 2600, 1)
        train_y: (2600, )
    """
    def encode_name(name):
        return 1 if name == target_name else 0
    # first remove id, id is the last column
    arr2d = arr2d.iloc[:, :-1]
    # because the data label is wrong
    arr2d = arr2d[arr2d.iloc[:, -1] != "answer"]
    # then split "answer" and other columns
    # "answer" is the last second column, the last one for now
    train_x = arr2d.iloc[:, :-1].astype(np.float).values
    # train_x = standardize(train_x, minmax=False).values

    train_y = arr2d.iloc[:, -1].apply(encode_name).astype(np.int8).values

    if remove_begin:
        train_x = train_x[:, num_remove_begin:]

    if remove_end:
        train_x = train_x[:, :-num_remove_end]

    if use_bin:
        train_x = average_bin_faster(train_x, num_bins=config.num_bins)

    train_x = train_x.reshape(*train_x.shape, 1)

    return train_x, train_y
