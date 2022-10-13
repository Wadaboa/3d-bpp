from __future__ import annotations

import itertools as it
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from .layers import Layer
from .plot import Coordinate, Cuboid, Dimension
from .superitems import SuperitemPool


def argsort(seq: Sequence[Any], reverse=False) -> list[int]:
    """Sort the given array and return indices instead of values."""
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def duplicate_keys(dicts: Iterable[dict[Hashable, Any]]) -> list[Hashable]:
    """Check that the input dictionaries have common keys."""
    keys = list(flatten([d.keys() for d in dicts]))
    return [k for k, v in Counter(keys).items() if v > 1]


def flatten(list_: Iterable) -> Iterable:
    """Given nested Python lists, return their flattened version."""
    for element in list_:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


def build_layer_from_model_output(
    superitems_pool, superitems_in_layer, solution, pallet_dims: Dimension
):
    """Return a single layer from the given model solution (either baseline or column generation).

    The 'solution' parameter should be a dictionary of the form
    {
        'c_{s}_x': ...,
        'c_{s}_y: ...,
        ...
    }
    """
    spool, scoords = [], []
    for s in superitems_in_layer:
        spool += [superitems_pool[s]]
        scoords += [Coordinate(x=solution[f"c_{s}_x"], y=solution[f"c_{s}_y"])]
    spool = SuperitemPool(superitems=spool)
    return Layer(spool, scoords, pallet_dims)


def do_overlap(a, b):
    """Check if the given items strictly overlap or not.

    (both items should be given as a Pandas Series)
    """
    assert isinstance(a, pd.Series) and isinstance(b, pd.Series), "Wrong input types"
    dx = min(a.x.item() + a.width.item(), b.x.item() + b.width.item()) - max(a.x.item(), b.x.item())
    dy = min(a.y.item() + a.depth.item(), b.y.item() + b.depth.item()) - max(a.y.item(), b.y.item())
    if (dx > 0) and (dy > 0):
        return True
    return False


def get_pallet_plot(pallet_dims: Dimension):
    """Compute an initial empty 3D-plot with the pallet dimensions."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("xkcd:white")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.text(0, 0, 0, "origin", size=10, zorder=1, color="k")
    ax.view_init(azim=60)
    ax.set_xlim3d(0, pallet_dims.width)
    ax.set_ylim3d(0, pallet_dims.depth)
    ax.set_zlim3d(0, pallet_dims.height)
    return ax


def plot_product(ax, item_id, coords, dims):
    """Add product to given axis."""
    cuboid = Cuboid(coords, dims)
    ax.scatter3D(cuboid.get_xs(), cuboid.get_ys(), cuboid.get_zs())
    ax.add_collection3d(
        Poly3DCollection(
            cuboid.to_faces(),
            facecolors=np.random.rand(1, 3),
            linewidths=1,
            edgecolors="r",
            alpha=0.45,
        )
    )
    center = cuboid.center
    ax.text(
        center.x,
        center.y,
        center.z,
        item_id,
        size=10,
        zorder=1,
        color="k",
    )
    return ax


def get_l0_lb(order, pallet_dims: Dimension):
    """L0 lower bound (aka continuos lower bound) for the minimum number of required bins.

    The worst case performance of this bound is 1 / 8.

    References:
        Silvano Martello, David Pisinger and Daniele Vigo,
        "The Three-Dimensional Bin Packing Problem",
        Operations Research, 1998.
    """
    return np.ceil(order.volume.sum() / pallet_dims.volume)


def get_l1_lb(order, pallet_dims: Dimension):
    """L1 lower bound for the minimum number of required bins.

    The worst-case performance of L1 can be arbitrarily bad.

    References:
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

    def get_l1j2(d1, bd1, d2, bd2, d3, bd3):
        j2 = get_j2(d1, bd1, d2, bd2)
        if len(j2) == 0:
            return 0.0
        ps = order[order[d3] <= bd3 / 2][d3].values
        max_ab = -np.inf
        for p in tqdm(ps):
            js = get_js(j2, p, d3, bd3)
            jl = get_jl(j2, p, d3, bd3)
            a = np.ceil((js[d3].sum() - (len(jl) * bd3 - jl[d3].sum())) / bd3)
            b = np.ceil((len(js) - (np.floor((bd3 - jl[d3].values) / p)).sum()) / np.floor(bd3 / p))
            max_ab = max(max_ab, a, b)

        return len(j2[j2[d3] > (bd3 / 2)]) + max_ab

    l1wh = get_l1j2(
        "width", pallet_dims.width, "height", pallet_dims.height, "depth", pallet_dims.depth
    )
    l1wd = get_l1j2(
        "width", pallet_dims.width, "depth", pallet_dims.depth, "height", pallet_dims.height
    )
    l1dh = get_l1j2(
        "depth", pallet_dims.depth, "width", pallet_dims.width, "height", pallet_dims.height
    )
    return max(l1wh, l1wd, l1dh), l1wh, l1wd, l1dh


def get_l2_lb(order, pallet_dims: Dimension):
    """L2 lower bound for the minimum number of required bins.

    The worst-case performance of L2 Lower bound is 2 / 3.

    References:
        Silvano Martello, David Pisinger and Daniele Vigo,
        "The Three-Dimensional Bin Packing Problem",
        Operations Research, 1998.
    """

    def get_kv(p, q, d1, bd1, d2, bd2):
        return order[(order[d1] > bd1 - p) & (order[d2] > bd2 - q)]

    def get_kl(kv, d1, bd1, d2, bd2):
        kl = order[~order.isin(kv)]
        return kl[(kl[d1] > (bd1 / 2)) & (kl[d2] > (bd2 / 2))]

    def get_ks(kv, kl, p, q, d1, d2):
        ks = order[~order.isin(pd.concat([kv, kl], axis=0))]
        return ks[(ks[d1] >= p) & (ks[d2] >= q)]

    def get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3):
        kv = get_kv(p, q, d1, bd1, d2, bd2)
        kl = get_kl(kv, d1, bd1, d2, bd2)
        ks = get_ks(kv, kl, p, q, d1, d2)

        return l1 + max(
            0,
            np.ceil(
                (pd.concat([kl, ks], axis=0).volume.sum() - (bd3 * l1 - kv[d3].sum()) * bd1 * bd2)
                / (bd1 * bd2 * bd3)
            ),
        )

    def get_l2j2(l1, d1, bd1, d2, bd2, d3, bd3):
        ps = order[(order[d1] <= bd1 // 2)][d1].values
        qs = order[(order[d2] <= bd2 // 2)][d2].values
        max_l2j2 = -np.inf
        for p, q in tqdm(it.product(ps, qs)):
            l2j2 = get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3)
            max_l2j2 = max(max_l2j2, l2j2)
        return max_l2j2

    _, l1wh, l1wd, l1hd = get_l1_lb(order, pallet_dims)
    l2wh = get_l2j2(
        l1wh, "width", pallet_dims.width, "height", pallet_dims.height, "depth", pallet_dims.depth
    )
    l2wd = get_l2j2(
        l1wd, "width", pallet_dims.width, "depth", pallet_dims.depth, "height", pallet_dims.height
    )
    l2dh = get_l2j2(
        l1hd, "depth", pallet_dims.depth, "height", pallet_dims.height, "width", pallet_dims.width
    )
    return max(l2wh, l2wd, l2dh), l2wh, l2wd, l2dh
