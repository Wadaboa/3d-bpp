from functools import partial


def select_layers(superitems, layer_pool, W, D, min_density=0.5, max_item_reps=3):
	densities = layer_pool.get_densities(superitems, W, D)
	sorted_layers = sorted(layer_pool, key=densities, reverse=True)
	sorted_layers = [l for d, l in zip(densities, sorted_layers) if d >= min_density]

def replace_items(superitems, layer_pool):
	for layer in layer_pool:
