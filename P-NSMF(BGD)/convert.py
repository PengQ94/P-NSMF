"""
Convert float to int, or int to float
"""

import numpy as np
import config as conf

u64 = np.iinfo(np.uint64)


def float2int(x):
    y = np.fix(x * (2 ** conf.prec_bits))
    y = np.where(y < 0, u64.max - np.uint64(abs(y)) + 1, np.uint64(y))
    return y


def int2float(y):
    x = np.where(
        y < 2 ** (conf.mask_bits - 1),
        np.float64(y) / 2 ** conf.prec_bits,
        -np.float64(-(y - u64.max - 1)) / 2 ** conf.prec_bits,
    )
    return x