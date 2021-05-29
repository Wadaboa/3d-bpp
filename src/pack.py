from functools import partial

def construct_bins(layer_pool, superitems, W, D, H, min_density=0.5):
	last_bin_index = 0
	bins = {last_bin_index: []}
	for layer in layer_pool.select_layers(superitems, W, D, min_density=min_density):
		placed = False
		for bin in bins:
			hb = sum(l.height for l in bins[bin])
			hl = layer.height
			if hb + hl <= H:
				bins[bin].insert(0, layer)
				placed = True

		if not placed:
			last_bin_index += 1
			bins[last_bin_index] = [layer]
	
	return bins
