import numpy as np
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf

from . import utils, layers, superitems


def get_height_groups(superitems_pool, height_tol=0):
    """
    Divide the whole pool of superitems into groups having either
    the exact same height or an height within the given tolerance
    """
    # Get unique heights
    unique_heights = sorted(set(s.height for s in superitems_pool))
    height_sets = {
        h: {k for k in unique_heights[i:] if k - h <= height_tol}
        for i, h in enumerate(unique_heights)
    }
    for (i, hi), (j, hj) in zip(list(height_sets.items())[:-1], list(height_sets.items())[1:]):
        if hj.issubset(hi):
            unique_heights.remove(j)

    # Generate one group of superitems for each similar height
    groups = []
    for height in unique_heights:
        pool = [
            s for s in superitems_pool if s.height >= height and s.height <= height + height_tol
        ]
        groups += [superitems.SuperitemPool(superitems=pool)]

    return groups


######################## TODO check never stopping recursion
def filter_height_groups(height_groups):
    def funzione(group, fringe, selected, subgroups=[]):
        if set(fringe) == set(range(len(group.superitems))):
            subgroups += [superitems.SuperitemPool(superitems=selected)]
            return subgroups

        for i, s in enumerate(group):
            if len(set(selected[-1].id).intersection(s.id)) > 0:
                fringe += [i]

        for i, s in enumerate(group):
            if i not in fringe:
                subgroups += funzione(group, fringe + [i], selected + [s], subgroups)

        return subgroups

    new_groups = []
    for group in height_groups:
        for i, s in enumerate(group):
            new_groups += funzione(group, [i], [s])

    return new_groups


def maxrects(superitems_pool, pallet_dims, add_single=True):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    return a layer pool with warm start placements
    """
    # Build a layer pool and return it if we are dealing
    # with a single superitem
    layer_pool = layers.LayerPool(superitems_pool, add_single=add_single)
    if len(superitems_pool) == 1:
        return layer_pool

    # Create the maxrects packing algorithm
    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=PackingBin.Global,
        pack_algo=MaxRectsBaf,
        sort_algo=SORT_AREA,
        rotation=False,
    )

    # Add an infinite number of layers (no upper bound)
    pallet_width, pallet_depth, _ = pallet_dims
    packer.add_bin(pallet_width, pallet_depth, count=float("inf"))

    # Add superitems to be packed
    ws, ds, _ = superitems_pool.get_superitems_dims()
    for i, (w, d) in enumerate(zip(ws, ds)):
        packer.add_rect(w, d, rid=i)

    # Start the packing procedure
    packer.pack()

    # Build a layer pool
    for layer in packer:
        spool = []
        scoords = []
        for superitem in layer:
            spool += [superitems_pool[superitem.rid]]
            scoords += [utils.Coordinate(superitem.x, superitem.y)]

        spool = superitems.SuperitemPool(superitems=spool)
        height = spool.get_max_height()
        layer_pool.add(layers.Layer(height, spool, scoords))

    return layer_pool
