import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA, SORT_SSIDE
from rectpack.maxrects import MaxRectsBaf, MaxRectsBl, MaxRectsBlsf, MaxRectsBssf

import utils, layers, superitems


def maxrects_multiple_layers(superitems_pool, pallet_dims, add_single=True):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    return a layer pool with warm start placements
    """
    logger.debug("Starting MR-ML-Offline")
    logger.debug(f"MR-ML-Offline: Used as warm_start -> {add_single}")
    logger.debug(f"MR-ML-Offline: Number of Superitems to place {len(superitems_pool)}")
    pack_algs = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]
    gen_lpools = []

    # Return a layer with a single item if only one
    # is present in the superitems pool
    if len(superitems_pool) == 1:
        layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=True)
    else:
        for alg in pack_algs:
            # Build initial layer pool
            layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=add_single)

            # Create the maxrects packing algorithm
            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.Global,
                pack_algo=alg,
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
                l_spool = []
                l_scoords = []
                for superitem in layer:
                    l_spool += [superitems_pool[superitem.rid]]
                    l_scoords += [utils.Coordinate(superitem.x, superitem.y)]

                l_spool = superitems.SuperitemPool(superitems=l_spool)
                l = layers.Layer(l_spool, l_scoords, pallet_dims)
                layer_pool.add(l)
                layer_pool.sort_by_densities(two_dims=False)
            gen_lpools += [layer_pool]

        # Find the best layerpool with most superitems placed and the less layers and more dense and return it
        n_notcovered = [len(lp.not_covered_superitems()) for lp in gen_lpools]
        n_layers = [len(lp) for lp in gen_lpools]
        denser_layer = [lp[0].get_density(two_dims=False) for lp in gen_lpools]
        lp_indexes = utils.argsort(list(zip(n_notcovered, n_layers, denser_layer)), reverse=True)
        layer_pool = gen_lpools[lp_indexes[0]]
        n_notcovered = n_notcovered[lp_indexes[0]]

    logger.debug(
        f"MR-Multiple-Layers: Generated layers {len(layer_pool)}, "
        f"Superitems placed {len(superitems_pool) - n_notcovered}/{len(superitems_pool)}\n"
        f"with 3D-densities {layer_pool.get_densities(two_dims=False)}"
    )
    # TODO Error in maxrects_multiple_layers Items in layer repetition, see test_order.cvs, required more testing
    # layer_pool.to_dataframe() used to proc get_items_dims and duplicate check
    return layer_pool


def maxrects_single_layer_offline(superitems_pool, pallet_dims, superitems_in_layer=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    try to fit each superitem in a single layer (if not possible, return an error)
    """
    logger.debug("Starting MR-SL-Offline")
    pack_algs = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]

    ws, ds, _ = superitems_pool.get_superitems_dims()
    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    logger.debug(
        f"MR-SL-Offline: Number of Superitems to place {superitems_in_layer}/{len(superitems_pool)}"
    )

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
            layer = layers.Layer(
                spool, [utils.Coordinate(s.x, s.y) for s in packer[0]], pallet_dims
            )
            logger.debug(
                f"MR-SL-Online: Generated new Layer with {len(layer)} Superitems, "
                f"with 3D-density {layer.get_density(two_dims=False)}"
            )
            return (True, layer)
    return False, None


def maxrects_single_layer_online(superitems_pool, pallet_dims, superitems_duals=None):
    """
    Given a SuperitemPool, the maximum dimensions to pack them into
    and and values from which to pick, try to fit following the given order,
    the greatest number of Superitems in a single Layer
    """
    logger.debug("Starting MR-SL-Online")
    pack_algs = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]

    ws, ds, hs = superitems_pool.get_superitems_dims()
    gen_layers, num_duals = [], []
    # If no duals are given use Superitems' height as dual,
    # casting to np.array to be coherent with usual duals format
    if superitems_duals is None:
        superitems_duals = np.array(hs)
    indexes = utils.argsort(superitems_duals, reverse=True)
    logger.debug(
        f"MR-SL-Online: Non-zero duals to place {sum(superitems_duals[i] > 0 for i in indexes)}"
    )
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
    n_duals = num_duals[layer_indexes[0]]
    logger.debug(
        f"MR-SL-Online: Generated new Layer with {len(layer)} Superitems, "
        f"of which {n_duals} with Dual > 0,"
        f"with 3D-density {layer.get_density(two_dims=False)}"
    )

    return layer
