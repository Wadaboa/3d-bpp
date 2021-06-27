from . import layers, superitems, config, cg, warm_start


def main(order, use_cg=True):
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool())
    bins_lbs = []

    working_order = order.copy()
    for i in range(3):
        superitems_pool = superitems.SuperitemPool(
            order=working_order, pallet_dims=config.PALLET_DIMS, max_vstacked=4, not_horizontal=True
        )

        height_groups = warm_start.get_height_groups(
            superitems_pool, config.PALLET_DIMS, height_tol=50, density_tol=0.5
        )

        for i, spool in enumerate(height_groups):
            print(f"Height group {i + 1}/{len(height_groups)}")
            layer_pool = warm_start.maxrects(spool, config.PALLET_DIMS, add_single=False)
            if use_cg:
                layer_pool, bins_lb = cg.column_generation(
                    layer_pool,
                    config.PALLET_DIMS,
                    max_iter=100,
                    max_stag_iters=5,
                    tlim=None,
                    use_maxrect=True,
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
