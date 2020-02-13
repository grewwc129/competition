import pandas as pd
from .config import config
from .maths import *
import numpy as np

global_x = np.arange(2600)


def standardize(df, minmax=True):
    if minmax:
        min_val, max_val = df.min(axis=0), df.max(axis=0)
        diff = max_val - min_val
        return (df - min_val) / diff

    mean, std = df.mean(axis=0), df.std(axis=0)
    return (df - mean) / std


def encode_name(name):
    try:
        return config.class_label[name]
    except:
        return -1000


def trim_df(df, use_bin=True):
    """
    use_bin:
        if True, use "average_bin" function
    returns:
        train_x: (None, 2600, 1)
        train_y: (2600, )
    """

    # first remove id, id is the last column
    df = df.iloc[:, :-1]
    df = df[df.iloc[:, -1] != "answer"]  # because the data label is wrong
    # then split "answer" and other columns
    # "answer" is the last second column, the last one for now
    train_x = df.iloc[:, :-1].astype(np.float).values
    # train_x = standardize(train_x, minmax=False).values

    train_y = df.iloc[:, -1].apply(encode_name).astype(np.int8).values
    if use_bin:
        train_x = np.array(
            [average_bin(global_x, x, num_bins=config.num_bins, normalize=True) for x in train_x])
    train_x = train_x.reshape(*train_x.shape, 1)
    return train_x, train_y


def getIthBatch(fname, start=0, nrows=1000, split_train_test=True):
    """the function is very slow when "start" is large
    """
    if start == 0:
        start += 1  # skip headers

    df = pd.read_csv(fname, engine='python', nrows=nrows,
                     skiprows=start, header=None)

    if not split_train_test:
        return df
    # first remove id, id is the last column
    df = df[df.columns[:-1]]

    # then split "answer" and other columns
    # "answer" is the last second column, the last one for now
    train_x = df[df.columns[:-1]].values
    train_y = df[df.columns[-1]
                 ].apply(lambda name: config.class_label[name]).values
    return train_x.reshape(*train_x.shape, 1), train_y
