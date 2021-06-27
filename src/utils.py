import itertools
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf

from . import superitems, layers


class Dimension:
    """
    Helper class to define object dimensions
    """

    def __init__(self, width, depth, height, weight=None):
        self.width = int(width)
        self.depth = int(depth)
        self.height = int(height)
        self.weight = int(weight)
        self.area = int(width * depth)
        self.volume = int(width * depth * height)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.width == other.width
                and self.depth == other.depth
                and self.height == other.height
                and self.weight == other.weight
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Dimension(width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Coordinate:
    """
    Helper class to define a pair/triplet of coordinates
    """

    def __init__(self, x, y, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def from_blb_to_vertices(self, dims):
        blb = self
        blf = Coordinate(self.x + dims.width, self.y, self.z)
        brb = Coordinate(self.x, self.y + dims.depth, self.z)
        brf = Coordinate(self.x + dims.width, self.y + dims.depth, self.z)
        tlb = Coordinate(self.x, self.y, self.z + dims.height)
        tlf = Coordinate(self.x + dims.width, self.y, self.z + dims.height)
        trb = Coordinate(self.x, self.y + dims.depth, self.z + dims.height)
        trf = Coordinate(self.x + dims.width, self.y + dims.depth, self.z + dims.height)
        return [blb, blf, brb, brf, tlb, tlf, trb, trf]

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()


class Vertices:
    def __init__(self, blb, dims):
        assert isinstance(blb, Coordinate)
        assert isinstance(dims, Dimension)
        self.dims = dims
        self.blb = blb
        self.blf = Coordinate(self.blb.x + self.dims.width, self.blb.y, self.blb.z)
        self.brb = Coordinate(self.blb.x, self.blb.y + self.dims.depth, self.blb.z)
        self.brf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y + self.dims.depth, self.blb.z
        )
        self.tlb = Coordinate(self.blb.x, self.blb.y, self.blb.z + self.dims.height)
        self.tlf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y, self.blb.z + self.dims.height
        )
        self.trb = Coordinate(
            self.blb.x, self.blb.y + self.dims.depth, self.blb.z + self.dims.height
        )
        self.trf = Coordinate(
            self.blb.x + self.dims.width,
            self.blb.y + self.dims.depth,
            self.blb.z + self.dims.height,
        )
        self.vertices = [
            self.blb,
            self.blf,
            self.brb,
            self.brf,
            self.tlb,
            self.tlf,
            self.trb,
            self.trf,
        ]

    def get_center(self):
        return Coordinate(
            self.blb.x + self.dims.width // 2,
            self.blb.y + self.dims.depth // 2,
            self.blb.z + self.dims.height // 2,
        )

    def get_xs(self):
        return np.array([v.x for v in self.vertices])

    def get_ys(self):
        return np.array([v.y for v in self.vertices])

    def get_zs(self):
        return np.array([v.z for v in self.vertices])

    def to_faces(self):
        return np.array(
            [
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.brb.to_numpy(),
                ],  # bottom
                [
                    self.tlb.to_numpy(),
                    self.tlf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # top
                [
                    self.blb.to_numpy(),
                    self.brb.to_numpy(),
                    self.trb.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # back
                [
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.tlf.to_numpy(),
                ],  # front
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.tlf.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # left
                [
                    self.brb.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # right
            ]
        )


def argsort(seq, reverse=False):
    """
    Sort the given array and return indices instead of values
    """
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def check_duplicate_keys(dicts, err_msg):
    """
    Assert that the input dictionaries have no common keys
    """
    assert not duplicate_keys(dicts), err_msg


def duplicate_keys(dicts):
    """
    Check that the input dictionaries have common keys
    """
    keys = list(flatten([d.keys() for d in dicts]))
    return len([k for k, v in Counter(keys).items() if v > 1]) > 0


def flatten(l):
    """
    Given nested Python lists, return their flattened version
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_l0_lb(order, pallet_dims):
    """
    L0 lower bound for the minimum number of required bins

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """
    W, D, H = pallet_dims
    return np.ceil(order.volume.sum() / (W * D * H))


def get_l1_lb(order, pallet_dims):
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
    W, D, H = pallet_dims
    l1wh = get_l1j2("width", W, "depth", D, "height", H)
    l1wd = get_l1j2("width", W, "height", H, "depth", D)
    l1hd = get_l1j2("depth", D, "height", H, "width", W)
    return max(l1wh, l1wd, l1hd), l1wh, l1wd, l1hd


def get_l2_lb(order, pallet_dims):
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
    W, D, H = pallet_dims
    _, l1wh, l1wd, l1hd = get_l1_lb(order, W, D, H)
    l2wh = get_l2j2(l1wh, "width", W, "depth", D, "height", H)
    l2wd = get_l2j2(l1wd, "width", W, "height", H, "depth", D)
    l2hd = get_l2j2(l1hd, "depth", D, "height", H, "width", W)
    return max(l2wh, l2wd, l2hd), l2wh, l2wd, l2hd


def maxrects_single_layer(superitems_pool, ws, ds, W, D, superitems_in_layer=None):
    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    # Create the maxrects packing algorithm
    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=PackingBin.Global,
        pack_algo=MaxRectsBaf,
        sort_algo=SORT_AREA,
        rotation=False,
    )

    # Add one bin representing one layer
    packer.add_bin(W, D, count=1)

    # Add superitems to be packed
    for i in superitems_in_layer:
        packer.add_rect(ws[i], ds[i], rid=i)

    # Start the packing procedure
    packer.pack()

    # Unfeasible packing
    if len(packer) == 0:
        return False, None

    # Feasible packing with a single layer
    layer = packer[0]
    spool = superitems.SuperitemPool(superitems=[superitems_pool[s.rid] for s in layer])
    height = spool.get_max_height()
    return True, layers.Layer(height, spool, [Coordinate(s.x, s.y) for s in layer])
