import numpy as np
from .config import config
import keras.backend as K


def f1(y_true, y_pred):
    def f1_helper(y_true, pred_cls, cls_name):
        epsilon = 1e-8
        if isinstance(cls_name, str):
            cls_name = config.class_label[cls_name]
        true_labels = (y_true == cls_name)
        pred_labels = (pred_cls == cls_name)
        TP = np.sum(true_labels == pred_labels)
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

    return (f1_star + f1_galaxy+f1_qso)/3



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
