from loguru import logger

import layers, superitems, config, maxrects, column_generation, baseline, bins


def get_height_groups(superitems_pool, pallet_dims, height_tol=0, density_tol=0.5):
    """
    Divide the whole pool of superitems into groups having either
    the exact same height or an height within the given tolerance
    """
    assert height_tol >= 0 and density_tol >= 0.0, "Tolerance parameters must be non-negative"

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
        spool = [
            s for s in superitems_pool if s.height >= height and s.height <= height + height_tol
        ]
        spool = superitems.SuperitemPool(superitems=spool)
        if (
            sum(s.volume for s in spool)
            >= density_tol * spool.get_max_height() * pallet_dims.width * pallet_dims.depth
        ):
            groups += [spool]

    return groups


def maxrects_warm_start(superitems_pool, height_tol=0, density_tol=0.5, add_single=True):
    """
    Compute the warm start layer pool from maxrects, by calling the
    maxrects procedure on each height group
    """
    logger.info("MR computing layers")

    # Compute height groups and initialize initial layer pool
    height_groups = get_height_groups(
        superitems_pool, config.PALLET_DIMS, height_tol=height_tol, density_tol=density_tol
    )
    # If no height groups are identified fallback to one group
    if len(height_groups) == 0:
        logger.debug(f"MR found no height groups, falling back to standard procedure")
        height_groups = [superitems_pool]
    # Initial empty layer pool
    mr_layer_pool = layers.LayerPool(superitems_pool, config.PALLET_DIMS)

    # Call maxrects for each height group and merge all the layer pools
    for i, spool in enumerate(height_groups):
        logger.info(f"MR processing height group {i + 1}/{len(height_groups)}")
        layer_pool = maxrects.maxrects_multiple_layers(
            spool, config.PALLET_DIMS, add_single=add_single
        )
        mr_layer_pool.extend(layer_pool)
    logger.info(f"MR generated {len(mr_layer_pool)} layers")

    # Return the final layer pool
    return mr_layer_pool


def cg(
    superitems_pool,
    height_tol=0,
    density_tol=0.5,
    use_height_groups=True,
    mr_warm_start=True,
    max_iters=100,
    max_stag_iters=20,
    tlim=5,
    sp_mr=False,
    sp_np_type="mip",
    sp_p_type="cp",
    return_only_last=False,
    enable_solver_output=False,
):
    """
    Generate layers by calling the column generation procedure
    """
    logger.info("CG computing layers")

    # Initialize final layer pool
    cg_layer_pool = layers.LayerPool(superitems_pool, config.PALLET_DIMS)

    # Process superitems all together or by dividing them into height groups
    if use_height_groups:
        height_groups = get_height_groups(
            superitems_pool,
            config.PALLET_DIMS,
            height_tol=height_tol,
            density_tol=density_tol,
        )
        # If no height groups are identified fallback to one group
        if len(height_groups) == 0:
            logger.debug(f"CG found no height groups, falling back to standard procedure")
            height_groups = [superitems_pool]
    else:
        height_groups = [superitems_pool]

    # Iterate over each height group (or the entire superitems pool)
    # and call the column generation procedure for each one
    for i, spool in enumerate(height_groups):
        logger.info(f"CG processing height group {i + 1}/{len(height_groups)}")

        # Use either the warm start given by maxrects (over height groups)
        # or a warm start comprised of one layer for each item
        if mr_warm_start:
            warm_start_layer_pool = maxrects_warm_start(
                spool, height_tol=height_tol, density_tol=density_tol
            )
        else:
            warm_start_layer_pool = layers.LayerPool(spool, config.PALLET_DIMS, add_single=True)

        # Call the column generation procedure
        layer_pool, _ = column_generation.column_generation(
            warm_start_layer_pool,
            config.PALLET_DIMS,
            max_iter=max_iters,
            max_stag_iters=max_stag_iters,
            tlim=tlim,
            sp_mr=sp_mr,
            sp_np_type=sp_np_type,
            sp_p_type=sp_p_type,
            return_only_last=return_only_last,
            enable_solver_output=enable_solver_output,
        )
        cg_layer_pool.extend(layer_pool)

    return cg_layer_pool


def main(
    order,
    procedure="cg",
    max_iters=1,
    superitems_horizontal=True,
    superitems_horizontal_type="two-width",
    superitems_max_vstacked=4,
    density_tol=0.5,
    filtering_two_dims=False,
    filtering_max_coverage_all=3,
    filtering_max_coverage_single=3,
    tlim=None,
    enable_solver_output=False,
    height_tol=0,
    cg_use_height_groups=True,
    cg_mr_warm_start=True,
    cg_max_iters=100,
    cg_max_stag_iters=20,
    cg_sp_mr=False,
    cg_sp_np_type="mip",
    cg_sp_p_type="cp",
    cg_return_only_last=False,
):
    """
    External interface to all the implemented solutions to solve 3D-BPP
    """
    assert max_iters > 0, "The number of maximum iteration must be > 0"
    assert procedure in ("mr", "bl", "cg"), "Unsupported procedure"

    logger.info(f"{procedure.upper()} procedure starting")

    # Create the final superitems pool and a copy of the order
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool(), config.PALLET_DIMS)
    working_order = order.copy()

    # Iterate the specified number of times in order to reduce
    # the number of uncovered items at each iteration
    not_covered, all_singles_removed = [], []
    for iter in range(max_iters):
        logger.info(f"{procedure.upper()} iteration {iter + 1}/{max_iters}")

        # Create the superitems pool and call the baseline procedure
        superitems_list, singles_removed = superitems.SuperitemPool.gen_superitems(
            order=working_order,
            pallet_dims=config.PALLET_DIMS,
            max_vstacked=superitems_max_vstacked,
            horizontal=superitems_horizontal,
            horizontal_type=superitems_horizontal_type,
        )
        superitems_pool = superitems.SuperitemPool(superitems=superitems_list)
        all_singles_removed += singles_removed

        # Call the right packing procedure
        if procedure == "bl":
            layer_pool = baseline.baseline(superitems_pool, config.PALLET_DIMS, tlim=tlim)
        elif procedure == "mr":
            layer_pool = maxrects_warm_start(
                superitems_pool, height_tol=height_tol, density_tol=density_tol, add_single=False
            )
        elif procedure == "cg":
            layer_pool = cg(
                superitems_pool,
                height_tol=height_tol,
                density_tol=density_tol,
                use_height_groups=cg_use_height_groups,
                mr_warm_start=cg_mr_warm_start,
                max_iters=cg_max_iters,
                max_stag_iters=cg_max_stag_iters,
                tlim=tlim,
                sp_mr=cg_sp_mr,
                sp_np_type=cg_sp_np_type,
                sp_p_type=cg_sp_p_type,
                return_only_last=cg_return_only_last,
                enable_solver_output=enable_solver_output,
            )

        # Filter layers based on the given parameters
        layer_pool = layer_pool.filter_layers(
            min_density=density_tol,
            two_dims=filtering_two_dims,
            max_coverage_all=filtering_max_coverage_all,
            max_coverage_single=filtering_max_coverage_single,
        )

        # Add only the filtered layers
        final_layer_pool.extend(layer_pool)

        # Compute the number of uncovered Items
        prev_not_covered = len(not_covered)
        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]
        logger.info(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")
        if len(not_covered) == prev_not_covered:
            logger.info("Stop iterating, no improvement from the previous iteration")
            break

        # Compute a new order composed of only not covered items
        working_order = order.iloc[not_covered].copy()

    # Build a pool of bins from the layer pool and compact
    # all layers in each bin to avoid having "flying" products
    bin_pool = bins.BinPool(
        final_layer_pool, config.PALLET_DIMS, singles_removed=set(all_singles_removed)
    )
    return bins.CompactBinPool(bin_pool)
