import numpy as np
from loguru import logger
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf, MaxRectsBl, MaxRectsBlsf, MaxRectsBssf

import utils, layers, superitems


MAXRECTS_PACKING_STRATEGIES = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]


def maxrects_multiple_layers(superitems_pool, pallet_dims, add_single=True):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    return a layer pool with warm start placements
    """
    logger.debug("MR-ML-Offline starting")
    logger.debug(f"MR-ML-Offline {'used' if add_single else 'not_used'} as warm_start")
    logger.debug(f"MR-ML-Offline {len(superitems_pool)} superitems to place")

    # Return a layer with a single item if only one is present in the superitems pool
    if len(superitems_pool) == 1:
        layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=True)
        uncovered = 0
    else:
        generated_pools = []
        for strategy in MAXRECTS_PACKING_STRATEGIES:
            # Build initial layer pool
            layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=add_single)

            # Create the maxrects packing algorithm
            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.Global,
                pack_algo=strategy,
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
                spool, scoords = [], []
                for superitem in layer:
                    spool += [superitems_pool[superitem.rid]]
                    scoords += [utils.Coordinate(superitem.x, superitem.y)]

                spool = superitems.SuperitemPool(superitems=spool)
                layer_pool.add(layers.Layer(spool, scoords, pallet_dims))
                layer_pool.sort_by_densities(two_dims=False)

            # Add the layer pool to the list of generated pools
            generated_pools += [layer_pool]

        # Find the best layer pool by considering the number of placed superitems,
        # the number of generated layers and the density of each layer dense
        uncovered = [len(pool.not_covered_superitems()) for pool in generated_pools]
        n_layers = [len(pool) for pool in generated_pools]
        densities = [pool[0].get_density(two_dims=False) for pool in generated_pools]
        pool_indexes = utils.argsort(list(zip(uncovered, n_layers, densities)), reverse=True)
        layer_pool = generated_pools[pool_indexes[0]]
        uncovered = uncovered[pool_indexes[0]]

    logger.debug(
        f"MR-ML-Offline generated {len(layer_pool)} layers with 3D densities {layer_pool.get_densities(two_dims=False)}"
    )
    logger.debug(
        f"MR-ML-Offline placed {len(superitems_pool) - uncovered}/{len(superitems_pool)} superitems"
    )
    return layer_pool


def maxrects_single_layer_offline(superitems_pool, pallet_dims, superitems_in_layer=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    try to fit each superitem in a single layer (if not possible, return an error)
    """
    logger.debug("MR-SL-Offline starting")

    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    logger.debug(f"MR-SL-Offline {superitems_in_layer}/{len(superitems_pool)} superitems to place")

    # Iterate over each placement strategy
    ws, ds, _ = superitems_pool.get_superitems_dims()
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Offline,
            bin_algo=PackingBin.Global,
            pack_algo=strategy,
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

        # Feasible packing with a single layer
        if len(packer) == 1 and len(packer[0]) == len(superitems_in_layer):
            spool = superitems.SuperitemPool(superitems=[superitems_pool[s.rid] for s in packer[0]])
            layer = layers.Layer(
                spool, [utils.Coordinate(s.x, s.y) for s in packer[0]], pallet_dims
            )
            logger.debug(
                f"MR-SL-Offline generated a new layer with {len(layer)} superitems "
                f"and {layer.get_density(two_dims=False)} 3D density"
            )
            return layer

    return None


def maxrects_single_layer_online(superitems_pool, pallet_dims, superitems_duals=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into, try to fit
    the greatest number of superitems in a single layer following the given order
    """
    logger.debug("MR-SL-Online starting")

    # If no duals are given use superitems' heights as a fallback
    ws, ds, hs = superitems_pool.get_superitems_dims()
    if superitems_duals is None:
        superitems_duals = np.array(hs)

    # Sort rectangles by duals
    indexes = utils.argsort(list(zip(superitems_duals, hs)), reverse=True)
    logger.debug(
        f"MR-SL-Online {sum(superitems_duals[i] > 0 for i in indexes)} non-zero duals to place"
    )

    # Iterate over each placement strategy
    generated_layers, num_duals = [], []
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Online,
            pack_algo=strategy,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.depth, count=1)

        # Online packing procedure
        n_packed, non_zero_packed, layer_height = 0, 0, 0
        for i in indexes:
            if superitems_duals[i] > 0 or hs[i] <= layer_height:
                packer.add_rect(ws[i], ds[i], i)
                if len(packer[0]) > n_packed:
                    n_packed = len(packer[0])
                    if superitems_duals[i] > 0:
                        non_zero_packed += 1
                    if hs[i] > layer_height:
                        layer_height = hs[i]
        num_duals += [non_zero_packed]

        # Build layer after packing
        spool, coords = [], []
        for s in packer[0]:
            spool += [superitems_pool[s.rid]]
            coords += [utils.Coordinate(s.x, s.y)]
        layer = layers.Layer(superitems.SuperitemPool(spool), coords, pallet_dims)
        generated_layers += [layer]

    # Find the best layer by taking into account the number of
    # placed superitems with non-zero duals and density
    layer_indexes = utils.argsort(
        [
            (duals, layer.get_density(two_dims=False))
            for duals, layer in zip(num_duals, generated_layers)
        ],
        reverse=True,
    )
    layer = generated_layers[layer_indexes[0]]

    logger.debug(
        f"MR-SL-Online generated a new layer with {len(layer)} superitems "
        f"(of which {num_duals[layer_indexes[0]]} with non-zero dual) "
        f"and {layer.get_density(two_dims=False)} 3D density"
    )
    return layer
