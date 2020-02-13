import numpy as np


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
    """not finished yet
    """
    epsilon = 1e-8
    total = num_bins * (len(y) // num_bins)
    y1, y2 = y[:total], y[total:]
    y1 = y1.reshape(num_bins, len(y) // num_bins)

    y1 = np.mean(y1, axis=1).ravel()
    y2 = np.mean(y2).ravel()
    return np.concatenate([y1, y2])
