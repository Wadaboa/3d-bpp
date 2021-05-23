from functools import partial


def layer_density(superitems, layer, W, D):
	layer_volume = sum(superitems.iloc[s].volume for s in layer[1])
	return layer_volume / W * D * layer[0]

def select_layers(superitems, layer_pool, W, D, min_density=0.5, max_item_reps=3):
	densities = map(layer_pool, partial(layer_density, superitems=superitems, W=W, D=D))
	sorted_layers = sorted(layer_pool, key=densities, reverse=True)
	sorted_layers = [l for d, l in zip(densities, sorted_layers) if d >= min_density]

def replace_items(superitems, layer_pool):
	for layer in layer_pool:
