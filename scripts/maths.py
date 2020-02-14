import numpy as np
from .management_utils import *


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
    num_each_bin = num_cols // num_bins
    total = num_bins * num_each_bin
    y1, y2 = y[:, :total], y[:, total:]
    y1 = y1.reshape(len(y1), num_bins, num_each_bin)

    y1 = np.mean(y1, axis=-1).reshape(len(y), -1)
    if total != num_cols:
        y2 = np.mean(y2, axis=1)
        y1[:, -1] = (y1[:, -1] + y2)/2

    # normalize
    y1 -= y1.mean(axis=1)[:, None]
    y1 /= (np.max(np.abs(y1), axis=1)[:, None] + 1e-8)
    return y1


def change_zero_to_mean(arr2d):
    arr1d = np.ravel(arr2d)
    abs_arr1d = np.abs(arr1d)
    abs_arr2d = abs_arr1d.reshape(*arr2d.shape)
    effective_mean = np.mean(arr1d[abs_arr1d > 1e-4])
    arr2d[abs_arr2d <= 1e-4] = effective_mean
    return arr2d


def remove_sharp(arr2d, threshold=0.5e4):
    std = np.std(arr2d, axis=1)[:, None]
    diff = (np.abs(arr2d) - std) / (std+1e-8)
    res = np.where(diff > threshold, -std, diff)
    return res + std
