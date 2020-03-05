import numpy as np
from .config import config
import keras.backend as K
from .config import config
from .maths import *
from .general_utils import *


def set_bad_to_qso(spectra2d, pred_cls_1d):
    _, bad_mask = find_bad(spectra2d, label=None,
                           num_consecutive=2000, return_mask=True)
    # print("here", bad_mask)
    pred_cls_1d[bad_mask] = config.class_label['qso']
    return None


def get_predict_cls(pred2d, target_name=None, threshold=None):
    """
    the main purpose of this function is to lower "qso" threshold
    there are 7248 qso in the validation set, however, we missed 450
    """
    pred_cls_normal = np.max(pred2d, axis=1)
    predicted_as_qso_mask = (pred_cls_normal == config.class_label[target_name])


def get_precision(ytrue, pred, dealwith_badspectra=False, spectra2d=None):
    # assert len(ytrue) == len(pred) and len(ytrue) > 0
    if isinstance(ytrue[0], str):
        ytrue = encode_names(ytrue)

    pred_cls = np.argmax(pred, axis=1)
    
    if dealwith_badspectra:
        if spectra2d is None:
            raise Exception("miss the original spectra")
    
        if len(spectra2d.shape) == 3:
            spectra2d = spectra2d.reshape(*spectra2d.shape[:-1])

        set_bad_to_qso(spectra2d, pred_cls)

    return sum(ytrue == pred_cls) / len(pred)


def f1(y_true, y_pred):
    def f1_helper(y_true, pred_cls, cls_name):
        epsilon = 1e-8
        if isinstance(cls_name, str):
            cls_name = config.class_label[cls_name]
        true_labels = (y_true == cls_name)
        pred_labels = (pred_cls == cls_name)
        TP = np.sum(true_labels * pred_labels)
        FP = np.sum(true_labels < pred_labels)
        FN = np.sum(true_labels > pred_labels)
        prec = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * (prec * recall) / (prec + recall + epsilon)
        return f1

    num_class = len(config.class_label)
    pred_cls = np.argmax(y_pred, axis=1)

    f1_star = f1_helper(y_true, pred_cls, 'star')
    f1_galaxy = f1_helper(y_true, pred_cls, 'galaxy')
    f1_qso = f1_helper(y_true, pred_cls, 'qso')

    return (f1_star + f1_galaxy + f1_qso)/3


# has some unknown bug now
def calc_f1(y_true, y_pred):
    # assert len(y_pred.shape) == 2
    num_class = len(config.class_label)
    pred_cls = K.argmax(y_pred, axis=1)

    f1_star = calc_target_f1(y_true, pred_cls, 'star')
    f1_galaxy = calc_target_f1(y_true, pred_cls, 'galaxy')
    f1_qso = calc_target_f1(y_true, pred_cls, 'qso')
    return (f1_star + f1_galaxy + f1_qso) / 3


def calc_target_f1(y_true, pred_cls, cls_name):
    if isinstance(cls_name, str):
        cls_name = config.class_label[cls_name]
    target_true = K.cast(K.equal(y_true, cls_name), 'float32')
    target_pred = K.cast(K.equal(pred_cls, cls_name), 'float32')
    target_TP = K.sum(target_true * target_pred)
    target_FN = K.sum(
        K.cast(K.greater(target_true, target_pred), 'float32'))
    target_TP_plus_FP = K.sum(target_pred) + K.epsilon()
    target_prec = target_TP / (target_TP_plus_FP + K.epsilon())
    target_recall = target_TP / (target_TP + target_FN + K.epsilon())

    return 2 * (target_prec*target_recall) / (target_prec + target_recall + K.epsilon())
