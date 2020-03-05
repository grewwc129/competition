import numpy as np
from .config import config
import keras.backend as K
import pandas as pd
from typing import List


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, lr)
    return model


def decay_lr(model, decay_rate=0.9):
    prev_lr = K.eval(model.optimizer.lr)
    K.set_value(model.optimizer.lr, prev_lr*decay_rate)


def encode_names(names: List[str])->List[int]:
    return np.array([config.class_label[name] for name in names])


def encode_name(name: str)->int:
    return config.class_label[name]


def drop_non_numeric(df: pd.DataFrame):
    return df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()]


class ArgumentException(Exception):
    pass
