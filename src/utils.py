from collections.abc import Iterable


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
