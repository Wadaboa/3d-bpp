import numpy as np
from matplotlib import pyplot as plt
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA, SORT_SSIDE
from rectpack.maxrects import MaxRectsBaf, MaxRectsBl, MaxRectsBlsf, MaxRectsBssf

import utils, layers, superitems


def maxrects_multiple_layers(superitems_pool, pallet_dims, add_single=True):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    return a layer pool with warm start placements
    """
    # Return a layer with a single item if only one
    # is present in the superitems pool
    if len(superitems_pool) == 1:
        return layers.LayerPool(superitems_pool, pallet_dims, add_single=True)

    # Build initial layer pool
    layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=add_single)

    # Create the maxrects packing algorithm
    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=PackingBin.Global,
        pack_algo=MaxRectsBaf,
        sort_algo=SORT_AREA,
        rotation=False,
    )

    # Add an infinite number of layers (no upper bound)
    packer.add_bin(pallet_dims.width, pallet_dims.depth, count=float("inf"))

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
        l = layers.Layer(spool, scoords, pallet_dims)
        layer_pool.add(l)

    return layer_pool


def maxrects_single_layer(superitems_pool, pallet_dims, superitems_in_layer=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    try to fit each superitem in a single layer (if not possible, return an error)
    """
    pack_algs = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]

    ws, ds, _ = superitems_pool.get_superitems_dims()

    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    for alg in pack_algs:

        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Offline,
            bin_algo=PackingBin.Global,
            pack_algo=alg,
            sort_algo=SORT_AREA,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.depth, count=1)

        # Add superitems to be packed
        for i in superitems_in_layer:
            packer.add_rect(ws[i], ds[i], rid=i)

        # Start the packing procedure
        packer.pack()

        if len(packer) == 1 and len(packer[0]) == len(superitems_in_layer):
            # Feasible packing with a single layer
            spool = superitems.SuperitemPool(superitems=[superitems_pool[s.rid] for s in packer[0]])
            return True, layers.Layer(
                spool, [utils.Coordinate(s.x, s.y) for s in packer[0]], pallet_dims
            )
    return False, None


def maxrects_single_layer_online(superitems_pool, pallet_dims, superitems_duals):
    """
    Given a superitems pool, the maximum dimensions to pack them into
    and and values from which to pick, try to fit following the givem order,
    the greatest number of superitems in a single layer
    """
    pack_algs = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]

    ws, ds, hs = superitems_pool.get_superitems_dims()
    gen_layers, num_duals = [], []

    indexes = utils.argsort(superitems_duals, reverse=True)
    print("Non-Zero Duals:", sum(superitems_duals[i] > 0 for i in indexes))
    # TODO sub-ordering of 0 duals to maximize layer density
    for alg in pack_algs:

        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Online,
            pack_algo=alg,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.depth, count=1)

        n_packed = 0
        non_zero_packed = 0
        layer_height = 0
        for i in indexes:
            if superitems_duals[i] > 0 or hs[i] <= layer_height:
                packer.add_rect(ws[i], ds[i], i)
                if len(packer[0]) > n_packed:
                    n_packed = len(packer[0])
                    # print("Placed", superitems_duals[i], hs[i])
                    if superitems_duals[i] > 0:
                        non_zero_packed += 1
                    if hs[i] > layer_height:
                        layer_height = hs[i]
                    # print(n_packed, layer_height)

        # Build layer after packing
        spool, coords = [], []
        for s in packer[0]:
            spool += [superitems_pool[s.rid]]
            coords += [utils.Coordinate(s.x, s.y)]
        num_duals += [non_zero_packed]
        layer = layers.Layer(superitems.SuperitemPool(spool), coords, pallet_dims)
        gen_layers += [layer]

    # Find the layer with most non-zero duals superitems placed and the most dense one and return it
    layer_indexes = utils.argsort(
        [(duals, layer.get_density(two_dims=False)) for duals, layer in zip(num_duals, gen_layers)],
        reverse=True,
    )
    layer = gen_layers[layer_indexes[0]]
    # layer.plot()
    # plt.show()
    print(layer.get_density(two_dims=False))
    print(num_duals[layer_indexes[0]])
    return gen_layers[layer_indexes[0]]
