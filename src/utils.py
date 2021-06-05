import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


def argsort(seq):
    """
    Sort the given array and return indices instead of values
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_liquid_volume(dims):
    """
    Return the liquid volume of a collection of items,
    given their dimensions
    """
    volume = 0
    for dim in dims:
        volume += dim[0] * dim[1] * dim[2]
    return volume / 1e9


def flatten(l):
    """
    Given nested Python lists, return their flattened version
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_l0_lb(order, W, D, H):
    """
    L0 lower bound for the minimum number of required bins

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """
    return np.ceil(order.volume.sum() / (W * D * H))


def get_l1_lb(order, W, D, H):
    """
    L1 lower bound for the minimum number of required bins

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """

    def get_j2(d1, bd1, d2, bd2):
        return order[(order[d1] > (bd1 / 2)) & (order[d2] > (bd2 / 2))]

    def get_js(j2, p, d, bd):
        return j2[(j2[d] >= p) & (j2[d] <= (bd / 2))]

    def get_jl(j2, p, d, bd):
        return j2[(j2[d] > (bd / 2)) & (j2[d] <= bd - p)]

    def get_l1j2(d1, bd1, d2, bd2, d3, db3):
        j2 = get_j2(d1, bd1, d2, bd2)
        max_ab = -np.inf
        for p in range(1, db3 // 2 + 1):
            js = get_js(j2, p, d3, db3)
            jl = get_jl(j2, p, d3, db3)
            a = np.ceil((js[d2].sum() - (len(jl) * db3 - jl[d2].sum())) / db3)
            b = np.ceil((len(js) - (np.floor((db3 - jl[d2].values) / p)).sum()) / np.floor(db3 / p))
            if max(a, b) > max_ab:
                max_ab = max(a, b)

        return len(j2[j2[d2] > (db3 / 2)]) + max_ab

    # The last two dimensions (D and H) are inverted, since
    # Martello specifies W x H x D, while we use W x D x H
    l1wh = get_l1j2("width", W, "depth", D, "height", H)
    l1wd = get_l1j2("width", W, "height", H, "depth", D)
    l1hd = get_l1j2("depth", D, "height", H, "width", W)
    return max(l1wh, l1wd, l1hd), l1wh, l1wd, l1hd


def get_l2_lb(order, W, D, H):
    """
    L2 lower bound for the minimum number of required bins

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """

    def get_kv(p, q, d1, db1, d2, db2):
        return order[(order[d1] > db1 - p) & (order[d2] > db2 - q)]

    def get_kl(kv, d1, db1, d2, db2):
        kl = order[~order.isin(kv)].dropna()
        return kl[(kl[d1] > (db1 / 2)) & (kl[d2] > (db2 / 2))]

    def get_ks(kv, kl, p, q, d1, d2):
        ks = order[~order.isin(pd.concat([kv, kl], axis=0))].dropna()
        return ks[(ks[d1] >= p) & (ks[d2] >= q)]

    def get_l2j2pq(p, q, l1, d1, db1, d2, db2, d3, db3):
        kv = get_kv(p, q, d1, db1, d2, db2)
        kl = get_kl(kv, d1, db1, d2, db2)
        ks = get_ks(kv, kl, p, q, d1, d2)

        return l1 + max(
            0,
            np.ceil(
                (pd.concat([kl, ks], axis=0).volume.sum() - (db3 * l1 - kv[d3].sum()) * db1 * db2)
                / (db1 * db2 * db3)
            ),
        )

    def get_l2j2(l1, d1, db1, d2, db2, d3, db3):
        max_l2j2 = -np.inf
        pq_combs = list(
            itertools.product(list(range(1, db1 // 2 + 1)), list(range(1, db2 // 2 + 1)))
        )
        for p, q in tqdm(pq_combs):
            l2j2 = get_l2j2pq(p, q, l1, d1, db1, d2, db2, d3, db3)
            if l2j2 > max_l2j2:
                max_l2j2 = l2j2
        return max_l2j2

    # The last two dimensions (D and H) are inverted, since
    # Martello specifies W x H x D, while we use W x D x H
    _, l1wh, l1wd, l1hd = get_l1_lb(order, W, D, H)
    l2wh = get_l2j2(l1wh, "width", W, "depth", D, "height", H)
    l2wd = get_l2j2(l1wd, "width", W, "height", H, "depth", D)
    l2hd = get_l2j2(l1hd, "depth", D, "height", H, "width", W)
    return max(l2wh, l2wd, l2hd), l2wh, l2wd, l2hd
