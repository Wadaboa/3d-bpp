from . import layers, superitems, config, maxrects, column_generation, baseline


def baseline_procedure(order, tlim=None):
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool())
    working_order = order.copy()
    for it in range(1):
        superitems_pool = superitems.SuperitemPool(
            order=working_order,
            pallet_dims=config.PALLET_DIMS,
            max_vstacked=4,
            not_horizontal=True,
        )

        layer_pool = baseline.call_baseline_model(superitems_pool, config.PALLET_DIMS, tlim=tlim)
        final_layer_pool.extend(layer_pool)

        final_layer_pool = final_layer_pool.select_layers(
            config.PALLET_WIDTH, config.PALLET_DEPTH, min_density=0.5, max_coverage=3
        )

        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]
        print(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")
        working_order = order.iloc[not_covered].copy()

    return final_layer_pool


def column_generation_procedure(order, use_cg=True, tlim=None):
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool())
    bins_lbs = []

    working_order = order.copy()
    for it in range(1):
        superitems_pool = superitems.SuperitemPool(
            order=working_order,
            pallet_dims=config.PALLET_DIMS,
            max_vstacked=4,
            not_horizontal=True,
        )

        height_groups = get_height_groups(
            superitems_pool, config.PALLET_DIMS, height_tol=50, density_tol=0.5
        )

        for i, spool in enumerate([height_groups[0]]):
            print(f"Height group {i + 1}/{len(height_groups)}")
            """
            layer_pool = maxrects.maxrects_multiple_layers(
                spool, config.PALLET_DIMS, add_single=False
            )
            """
            layer_pool = layers.LayerPool(spool, add_single=True)
            display(layer_pool.to_dataframe())
            if use_cg:
                layer_pool, bins_lb = column_generation.column_generation(
                    layer_pool,
                    config.PALLET_DIMS,
                    max_iter=100,
                    max_stag_iters=20,
                    tlim=tlim,
                    use_maxrect=True,
                    return_warm_start=False,
                )
                bins_lbs.append(bins_lb)
            final_layer_pool.extend(layer_pool)

        # Add single item superitems that were discarded by
        # the initial height groups procedure
        final_layer_pool.extend_superitems_pool(superitems_pool)

        final_layer_pool = final_layer_pool.select_layers(
            config.PALLET_WIDTH, config.PALLET_DEPTH, min_density=0.5, max_coverage=3
        )

        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]
        print(f"Items not covered: {len(not_covered)}/{len(item_coverage)}")
        working_order = order.iloc[not_covered].copy()

    return final_layer_pool, bins_lbs


def get_height_groups(superitems_pool, pallet_dims, height_tol=0, density_tol=0.5):
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
        spool = [
            s for s in superitems_pool if s.height >= height and s.height <= height + height_tol
        ]
        spool = superitems.SuperitemPool(superitems=spool)
        pallet_width, pallet_depth, _ = pallet_dims
        if (
            sum(s.volume for s in spool)
            >= density_tol * spool.get_max_height() * pallet_width * pallet_depth
        ):
            groups += [spool]

    return groups


if __name__ == "__main__":
    pass
