from functools import partial


def construct_bins(layer_pool, pallet_dims, min_density=0.5, two_dims=True, max_coverage=3):
    W, D, H = pallet_dims
    last_bin_index = 0
    bins = {last_bin_index: []}

    selected_layers = layer_pool.select_layers(
        W, D, min_density=min_density, two_dims=two_dims, max_coverage=max_coverage
    )
    print(selected_layers)

    for layer in selected_layers:
        placed = False
        for bin in bins:
            hb = sum(l.height for l in bins[bin])
            hl = layer.height
            if hb + hl <= H:
                bins[bin].append(layer)
                # bins[bin].insert(0, layer)
                placed = True

        if not placed:
            last_bin_index += 1
            bins[last_bin_index] = [layer]

    return bins
