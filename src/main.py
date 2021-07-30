from loguru import logger

import layers, superitems, config, maxrects, column_generation, baseline


def get_height_groups(superitems_pool, pallet_dims, height_tol=0, density_tol=0.5):
    """
    Divide the whole pool of superitems into groups having either
    the exact same height or an height within the given tolerance
    """
    assert height_tol >= 0 and density_tol >= 0.0, "Tollerances parameters must be non-negative"
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
    logger.info("Computing MR layers")

    # Compute height groups and initialize initial layer pool
    height_groups = get_height_groups(
        superitems_pool, config.PALLET_DIMS, height_tol=height_tol, density_tol=density_tol
    )
    # If no height groups are identified fallback to one group
    if len(height_groups) == 0:
        logger.debug(f"No height groups found, fallback to standard procedure")
        height_groups = [superitems_pool]
    # Initial empty layer pool
    mr_layer_pool = layers.LayerPool(superitems_pool, config.PALLET_DIMS)

    # Call maxrects for each height group and merge all the layer pools
    for i, spool in enumerate(height_groups):
        logger.info(f"Processing height group {i + 1}/{len(height_groups)}")
        layer_pool = maxrects.maxrects_multiple_layers(
            spool, config.PALLET_DIMS, add_single=add_single
        )
        mr_layer_pool.extend(layer_pool)
    logger.info(f"Generated {len(mr_layer_pool)} layers in MR")

    # Return the final layer pool
    return mr_layer_pool


# TODO all the 3 following procedure are basically the same, only the core function varies,
# create a generalization with a parameter


def baseline_procedure(
    order,
    max_iters=1,
    density_tol=0.5,
    filtering_two_dims=False,
    filtering_max_coverage_all=3,
    filtering_max_coverage_single=3,
    filtering_remove_duplicated=True,
    baseline_tlim=None,
):
    """
    Generate layers by calling the full-on mathematical programming approach
    (with layers)
    """
    assert max_iters > 0, "The number of maximum iteration must be > 0"

    logger.info("Starting Baseline procedure")
    # Create the final superitems pool and a copy of the order
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool(), config.PALLET_DIMS)
    working_order = order.copy()

    # Iterate the specified number of times in order to reduce
    # the number of uncovered items at each iteration
    # TODO add a relaxation in the filtering procedure based on the number of remaing items,
    # or the increment in number of iteration is pointless
    for iter in range(max_iters):
        logger.info(f"Baseline iteration {iter + 1}/{max_iters}")

        # Create the superitems pool and call the baseline procedure
        superitems_pool = superitems.SuperitemPool(
            superitems=superitems.SuperitemPool.gen_superitems(
                order=working_order,
                pallet_dims=config.PALLET_DIMS,
                max_vstacked=4,
                not_horizontal=True,
            )
        )
        layer_pool = baseline.baseline(superitems_pool, config.PALLET_DIMS, tlim=baseline_tlim)

        # Filter Layers based on the given parameters
        layer_pool = layer_pool.select_layers(
            min_density=density_tol,
            two_dims=filtering_two_dims,
            max_coverage_all=filtering_max_coverage_all,
            max_coverage_single=filtering_max_coverage_single,
            remove_duplicated=filtering_remove_duplicated,
        )

        # Add only the filtered Layers
        final_layer_pool.extend(layer_pool)

        # Compute the number of uncovered Items
        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]
        # Compute a new order compose of only not covered Items
        working_order = order.iloc[not_covered].copy()
        logger.info(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")

    return final_layer_pool


def maxrects_procedure(
    order,
    max_iters=1,
    height_tol=0,
    density_tol=0.5,
    filtering_two_dims=False,
    filtering_max_coverage_all=3,
    filtering_max_coverage_single=3,
    filtering_remove_duplicated=True,
):
    """
    Generate layers by calling maxrects on each height group and
    merge everything in a single pool
    """
    assert max_iters > 0, "The number of maximum iteration must be > 0"

    logger.info("Starting Maxrects procedure")
    # Create the final superitems pool and a copy of the order
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool(), config.PALLET_DIMS)
    working_order = order.copy()

    # Iterate the specified number of times in order to reduce
    # the number of uncovered items at each iteration
    # TODO add a relaxation on the height groups min_density and
    # in the filtering procedure based on the number of remaing items,
    # or the increment in number of iteration is pointless
    for iter in range(max_iters):
        logger.info(f"Maxrects iteration {iter + 1}/{max_iters}")

        # Create the superitems pool and call the maxrects procedure
        superitems_pool = superitems.SuperitemPool(
            superitems=superitems.SuperitemPool.gen_superitems(
                order=working_order,
                pallet_dims=config.PALLET_DIMS,
                max_vstacked=4,
                not_horizontal=True,
            )
        )

        layer_pool = maxrects_warm_start(
            superitems_pool, height_tol=height_tol, density_tol=density_tol, add_single=False
        )

        # Filter Layers based on the given parameters
        layer_pool = layer_pool.select_layers(
            min_density=density_tol,
            two_dims=filtering_two_dims,
            max_coverage_all=filtering_max_coverage_all,
            max_coverage_single=filtering_max_coverage_single,
            remove_duplicated=filtering_remove_duplicated,
        )
        logger.info(f"Number of filtered layers {len(layer_pool)}")

        # Add only the filtered Layers
        final_layer_pool.extend(layer_pool)

        # Compute the number of uncovered Items
        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]
        # Compute a new order compose of only not covered Items
        working_order = order.iloc[not_covered].copy()
        logger.info(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")

    logger.success(f"Final number of layers {len(final_layer_pool)}")
    # Return the final layer pool
    return final_layer_pool


def column_generation_procedure(
    order,
    use_height_groups=True,
    max_iters=1,
    height_tol=0,
    density_tol=0.5,
    mr_warm_start=True,
    cg_max_iters=100,
    cg_max_stag_iters=20,
    cg_tlim=5,
    filtering_two_dims=False,
    filtering_max_coverage_all=3,
    filtering_max_coverage_single=3,
    filtering_remove_duplicated=True,
):
    """
    Generate layers by calling the column generation procedure
    """
    assert max_iters > 0, "The number of maximum iteration must be > 0"
    logger.info("Starting Column generation procedure")
    # Create the final LayerPool and a copy of the order
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool(), config.PALLET_DIMS)
    working_order = order.copy()

    # Iterate the specified number of times in order to reduce
    # the number of uncovered items at each iteration
    # TODO add a relaxation on the height groups min_density and
    # in the filtering procedure based on the number of remaing items,
    # or the increment in number of iteration is pointless
    for iter in range(max_iters):
        logger.info(f"CG iteration {iter + 1}/{max_iters}")

        # Create the superitems pool
        superitems_pool = superitems.SuperitemPool(
            superitems=superitems.SuperitemPool.gen_superitems(
                order=working_order,
                pallet_dims=config.PALLET_DIMS,
                max_vstacked=4,
                not_horizontal=True,
                only_single=True,
            )
        )

        # Process superitems all together or by dividing them into height groups
        if use_height_groups:
            height_groups = get_height_groups(
                superitems_pool, config.PALLET_DIMS, height_tol=height_tol, density_tol=density_tol
            )
        else:
            height_groups = [superitems_pool]

        # Iterate over each height group (or the entire superitems pool)
        # and call the column generation procedure for each one
        bins_lbs = []
        for i, spool in enumerate(height_groups):
            logger.info(f"Processing height group {i + 1}/{len(height_groups)}")

            # Use either the warm start given by maxrects (over height groups)
            # or a warm start comprised of one layer for each item
            if mr_warm_start:
                warm_start_layer_pool = maxrects_warm_start(
                    spool, height_tol=height_tol, density_tol=density_tol
                )
            else:
                warm_start_layer_pool = layers.LayerPool(spool, config.PALLET_DIMS, add_single=True)

            # Call the column generation procedure
            layer_pool, bins_lb = column_generation.column_generation(
                warm_start_layer_pool,
                config.PALLET_DIMS,
                max_iter=cg_max_iters,
                max_stag_iters=cg_max_stag_iters,
                tlim=cg_tlim,
                spp_mr=False,
                spp_cp=True,
                spp_mip=False,
                pricing_problem_maxrect=False,
                return_only_last=False,
            )

            # Filter layers based on the given parameters
            layer_pool = layer_pool.select_layers(
                min_density=density_tol,
                two_dims=filtering_two_dims,
                max_coverage_all=filtering_max_coverage_all,
                max_coverage_single=filtering_max_coverage_single,
                remove_duplicated=filtering_remove_duplicated,
            )

            logger.info(f"Number of filtered layers {len(layer_pool)}")

            # Add only the filtered Layers and the Bins lower bounds
            bins_lbs.append(bins_lb)
            final_layer_pool.extend(layer_pool)

            # Compute the number of uncovered items
            item_coverage = layer_pool.item_coverage()
            not_covered = [k for k, v in item_coverage.items() if not v]
            # Compute a new order compose of only not covered Items
            working_order = order.iloc[not_covered].copy()
            logger.info(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")

    return layer_pool, bins_lbs
