"""
Generate random numbers
"""

import numpy as np
from numpy.random import Generator, PCG64

u64 = np.iinfo(np.uint64)
u8 = np.iinfo(np.uint8)


def generate_mask(dimension):
    rng = Generator(PCG64())
    mask = rng.integers(u64.max, size=dimension, dtype=np.uint64, endpoint=True)
    return mask


def generate_positiveInt8(dimension):
    rng = Generator(PCG64())
    return rng.integers(1, u8.max, size=dimension, dtype=np.uint64, endpoint=True)
