import numpy as np
from .config import config
import keras.backend as K


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, lr)
    return model


def decay_lr(model, decay_rate=0.9):
    prev_lr = K.eval(model.optimizer.lr)
    K.set_value(model.optimizer.lr, prev_lr*decay_rate)


def encode_names(names):
    return np.array([config.class_label[name] for name in names])


def get_precision(ytrue, pred):
    # assert len(ytrue) == len(pred) and len(ytrue) > 0
    if isinstance(ytrue[0], str):
        ytrue = encode_names(ytrue)

    return sum(ytrue == np.argmax(pred, axis=1)) / len(pred)
