import numpy as np
from .management_utils import *
import gc 

# @Deperacated
def median_bin(x, y, num_bins, bin_width=None, normalize=True):
    """
    assume x is sorted 
    bin_width is the scale factor of (x_max-x_min)/num_bins, not the absolute value
    """
    x_min, x_max = x[0], x[-1]
    default_bin_width = (x_max - x_min) / num_bins
    bin_width = default_bin_width if bin_width is None else bin_width
    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    if bin_spacing < 0:
        raise ValueError(
            f'bin space {bin_spacing} < 0 (maybe "bin_width" is too large)')

    res = np.repeat(np.median(y), num_bins)
    bin_lo, bin_hi = x_min, x_min + bin_width

    for i in range(num_bins):
        indices = np.argwhere(
            (x < bin_hi) & (x >= bin_lo)).ravel()

        if len(indices) > 0:
            res[i] = np.median(y[indices])

        bin_lo += bin_spacing
        bin_hi += bin_spacing

    if normalize:
        res -= np.median(res)
        res /= (np.max(np.abs(res)) + 1e-8)

    return res


# @Deperacated
def average_bin(x, y, num_bins, bin_width=None, normalize=True):
    """
    assume x is sorted 
    bin_width is the scale factor of (x_max-x_min)/num_bins, not the absolute value
    """
    epsilon = 1e-8
    x_min, x_max = x[0], x[-1]
    default_bin_width = (x_max - x_min) / num_bins
    bin_width = default_bin_width if bin_width is None else bin_width
    bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

    if bin_spacing < 0:
        raise ValueError(
            f'bin space {bin_spacing} < 0 (maybe "bin_width" is too large)')

    res = np.repeat(np.mean(y), num_bins)
    bin_lo, bin_hi = x_min, x_min + bin_width

    for i in range(num_bins):
        indices = np.argwhere(
            (x < bin_hi) & (x >= bin_lo)).ravel()

        if len(indices) > 0:
            res[i] = np.mean(y[indices])

        bin_lo += bin_spacing
        bin_hi += bin_spacing

    if normalize:
        res -= np.mean(res)
        res /= np.max(np.abs(res)+epsilon)

    return res


def average_bin_faster(y, num_bins):
    num_cols = len(y[0])
    num_rows = len(y)
    num_each_bin = num_cols // num_bins
    total = num_bins * num_each_bin
    y1, y2 = y[:, :total], y[:, total:]

    del y 
    gc.collect()

    y1 = y1.reshape(len(y1), num_bins, num_each_bin)

    y1 = np.mean(y1, axis=-1).reshape(num_rows, -1)
    if total != num_cols:
        y2 = np.mean(y2, axis=1)
        y1[:, -1] = (y1[:, -1] + y2)/2

    # normalize
    y1 -= y1.mean(axis=1)[:, None]
    y1 /= (np.max(np.abs(y1), axis=1)[:, None] + 1e-8)
    return y1


def median_bin_faster(y, num_bins):
    num_cols = len(y[0])
    num_each_bin = num_cols // num_bins
    total = num_bins * num_each_bin
    y1, y2 = y[:, :total], y[:, total:]
    y1 = y1.reshape(len(y1), num_bins, num_each_bin)

    y1 = np.median(y1, axis=-1).reshape(len(y), -1)
    if total != num_cols:
        y2 = np.median(y2, axis=1)
        y1[:, -1] = (y1[:, -1] + y2)/2

    # normalize
    y1 -= np.median(y1, axis=1)[:, None]
    y1 /= (np.max(np.abs(y1), axis=1)[:, None] + 1e-8)
    return y1


def change_zero_to_mean(arr2d):
    arr1d = np.ravel(arr2d)
    abs_arr1d = np.abs(arr1d)
    abs_arr2d = abs_arr1d.reshape(*arr2d.shape)
    effective_mean = np.mean(arr1d[abs_arr1d > 1e-4])
    arr2d[abs_arr2d <= 1e-4] = effective_mean
    return arr2d


def remove_sharp(arr2d, threshold=3):
    """
    bad
    """
    std = np.std(arr2d, axis=1)[:, None]
    mean = np.mean(arr2d, axis=1)[:, None]
    threshold *= std
    arr2d -= mean
    diff = np.abs(arr2d)
    res = np.where(diff > threshold, 0, arr2d)
    return res + mean


def find_bad(arr2d, label, return_mask=False, num_consecutive=2000):
    """
    find the bad spectral in arr2d
    returns: 
        if return_bad:
            good_spectra, bad_spectra
        else:
            good_spectra, good_labels
            good_labels means corresponding labels of good_spectra
    """

    diff = arr2d[:, 1:] - arr2d[:, :-1]

    #  if more than 2000 pairs of similar neighbor points, remove the dataset
    good_mask = np.sum(np.abs(diff) < 1e-8, axis=1) < num_consecutive

    if return_mask:
        return arr2d[good_mask], np.logical_not(good_mask)
    else:
        return arr2d[good_mask], label[good_mask]


def remove_badpoints_and_normalize(arr2d):
    """
    @Author: Yushan Li
    """
    mean = np.mean(arr2d, axis=1, keepdims=True)
    std = np.std(arr2d, axis=1, ddof=1, keepdims=True)
    std_20 = std*20
    arr2d = np.where((arr2d > mean+std_20) | (arr2d < mean-std_20), mean, arr2d)

    gc.collect()

    mean = np.mean(arr2d, axis=1, keepdims=True)
    std = np.std(arr2d, axis=1, ddof=1, keepdims=True)
    std_5 = std*5

    del std 
    gc.collect()

    lo, hi = mean - std_5, mean + std_5
    arr2d = np.where(arr2d > hi, hi, arr2d)
    arr2d = np.where(arr2d < lo, lo, arr2d)
    arr2d = np.where(arr2d == 0, mean, arr2d)

    gc.collect()

    min_val, max_val = np.min(arr2d, axis=1, keepdims=True), np.max(
        arr2d, axis=1, keepdims=True)
    arr2d = (arr2d - min_val) / (max_val - min_val + 1e-8)
    return arr2d - np.median(arr2d, axis=1, keepdims=True)
