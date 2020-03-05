import pandas as pd
from .config import config
from .maths import *
import numpy as np
from .general_utils import drop_non_numeric, encode_name
from .csv_to_h5 import read_h5
import gc 


def standardize(arr2d, minmax=True):
    if minmax:
        min_val, max_val = arr2d.min(axis=1), arr2d.max(axis=1)
        diff = max_val - min_val
        return (arr2d - min_val) / (diff+1e-8)

    mean, std = arr2d.mean(axis=1), arr2d.std(axis=1)
    return (arr2d - mean) / (std+1e-8)


def load_training(filename: str = config.train_file_name,
                  use_bin=True,
                  remove_endpoints=True):
    """
    use_bin:
        if True, use "average_bin" function
    returns:
        train_x: (None, Dim, 1)
        train_y: (Dim, )
    """
    arr2d = read_h5(filename)
    
    num_remove_begin = 50
    num_remove_end = 1

    train_x = arr2d[:, :-1]
    train_y = arr2d[:, -1]

    del arr2d
    gc.collect()

    #################################################################
    #################################################################
    #  preprocessing                                                #
    #################################################################
    if remove_endpoints:
        train_x = train_x[:, num_remove_begin:-num_remove_end]
    # remove bad points
    # train_x, train_y = find_bad(train_x, train_y, num_consecutive=2000)

    train_x = remove_badpoints_and_normalize(train_x)
    if use_bin:
        train_x = average_bin_faster(train_x, num_bins=config.num_bins)

    gc.collect()
    #################################################################
    #################################################################

    train_x = train_x.reshape(*train_x.shape, 1)

    gc.collect()
    
    return train_x, train_y


def trim_df_binary(arr2d, target_name, use_bin=True, remove_endpoints=True):
    """
    use_bin:
        if True, use "average_bin" function
    returns:
        train_x: (None, 2600, 1)
        train_y: (2600, )
    """
    num_remove_begin = 50
    num_remove_end = 1

    def encode_name(name):
        return 1 if name == target_name else 0
    # first remove id, id is the last column
    arr2d = arr2d.iloc[:, :-1]
    # because the data label is wrong
    arr2d = arr2d[arr2d.iloc[:, -1] != "answer"]
    # then split "answer" and other columns
    # "answer" is the last second column, the last one for now
    train_x = arr2d.iloc[:, :-1].astype(np.float).values

    train_y = arr2d.iloc[:, -1].apply(encode_name).astype(np.int8).values

    if remove_endpoints:
        train_x = train_x[:, num_remove_begin:-num_remove_end]

    if use_bin:
        train_x = median_bin_faster(train_x, num_bins=config.num_bins)

    # remove bad points
    train_x, train_y = find_bad(train_x, train_y, num_consecutive=2000)

    train_x = remove_badpoints_and_normalize(train_x)

    train_x = train_x.reshape(*train_x.shape, 1)

    return train_x, train_y
