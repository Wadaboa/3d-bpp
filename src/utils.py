from collections.abc import Iterable

import numpy as np


def get_liquid_volume(dims):
    volume = 0
    for dim in dims:
        volume += (dim[0] * dim[1] * dim[2])
    return volume / 1e+9


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def np_are_equal(a1, a2):
    if a1.shape != a2.shape:
        return False
    return np.count_nonzero(a1 - a2) == 0
