import numpy as np
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf

from . import utils, layers


def get_initial_groups(superitems, tol=0): 
    unique_heights = sorted(superitems.height.unique())
    height_sets = {
        h: {k for k in unique_heights[i:] if k - h <= tol}
        for i, h in enumerate(unique_heights)
    }
    for (i, hi), (j, hj) in zip(list(height_sets.items())[:-1], list(height_sets.items())[1:]):
        if hj.issubset(hi):
            unique_heights.remove(j)
    
    groups = []
    for height in unique_heights:
        groups += [
            superitems[
                (superitems.height >= height) &  
                (superitems.height <= height + tol)
            ].copy()
        ]

    final_groups = []
    for group in groups:
        sorted_group = group.sort_values(
            by=["num_items", "vstacked"], ascending=False
        ).reset_index().rename(columns={"index": "superitem_id"})
        indexes_to_remove = []
        for i, row in sorted_group.iterrows():
            for item in row["flattened_items"]:
                to_remove = sorted_group[
                    (sorted_group["flattened_items"].map(lambda l: item in l)) &
                    (sorted_group.index > i)
                ]
                indexes_to_remove += to_remove.index.tolist()

        final_groups += [sorted_group.drop(index=indexes_to_remove).reset_index(drop=True)]

    return final_groups


def maxrects(rects, bin_lenght, bin_width):
	packer = newPacker(
		mode=PackingMode.Offline, 
		bin_algo=PackingBin.Global, 
		pack_algo=MaxRectsBaf, 
		sort_algo=SORT_AREA, 
		rotation=False
	)

	packer.add_bin(
		bin_lenght, bin_width, count=float("inf")
	)

	for rect in rects:
		packer.add_rect(*rect)

	packer.pack()

	bins = []
	for bin_id in range(len(packer)):
		rects_in_bin = []
		for rect_id in range(len(packer[bin_id])):
			rects_in_bin += [(
				packer[bin_id][rect_id].rid,
				packer[bin_id][rect_id].x,
				packer[bin_id][rect_id].y
			)]
		bins.append(rects_in_bin)
	
	return bins


def single_item_layers(group):
	ol = np.zeros((len(group), ), dtype=int)
	zsl = np.eye(len(group), dtype=int)
	for i, item in group.iterrows():
		ol[i] = item.height
	return ol, zsl


def warm_start_groups(groups, num_total_superitems, pallet_lenght, pallet_width, add_single=True, split_by_group=True):
	'''
	Given a list of DataFrames of superitems, return the layer heights
	and superitems to layer assignments for each group or all the groups, 
	based on the maxrects algorithm
	'''
	groups_info = []
	layer_pool = layers.LayerPool()
	superitem_heights = dict()
	for group in groups:
		rects = []
		rect_ids = dict()
		for i, row in group.iterrows():
			rects += [(row.lenght, row.width, row.superitem_id)]
			rect_ids[row.superitem_id] = i
			superitem_heights[row.superitem_id] = row.height

		initial_layers = maxrects(rects, pallet_lenght, pallet_width)

		num_superitems = num_total_superitems if not split_by_group else len(rects) 
		zsl = np.zeros((num_superitems, len(initial_layers)), dtype=int)
		ol = np.zeros((len(initial_layers), ), dtype=int)
		for l, layer in enumerate(initial_layers):
			ids, coords = [], []
			for rect in layer:
				s = rect_ids[rect[0]] if split_by_group else rect[0]
				ids += [rect[0]]
				zsl[s, l] = 1
				coords += [(rect[1], rect[2])]
			ol[l] = group[group.superitem_id.isin(ids)].height.max()
			layer_pool.add(
				layers.Layer(ol[l], ids, np.array(coords))
			)

		inv_rect_ids = {v:k for k, v in rect_ids.items()}
		if add_single and len(group) > 1 and split_by_group:
			ol_single, zsl_single = single_item_layers(group)
			zsl = np.concatenate((zsl, zsl_single), axis=1)
			ol = np.concatenate((ol, ol_single))
			for i, h in enumerate(ol_single):
				layer_pool.add(
					layers.Layer(h, [inv_rect_ids[i]], np.array([(0, 0)]))
				)

		groups_info += [{'zsl': zsl, 'ol': ol, 'ids': rect_ids}]

	if not split_by_group:
		groups_info = [{
			'zsl': np.concatenate([g['zsl'] for g in groups_info], axis=1),
			'ol': np.concatenate([g['ol'] for g in groups_info]),
			'ids': {k: v for g in groups_info for k, v in g['ids'].items()},
		}]
		if add_single:
			groups_info[0]['zsl'] = np.concatenate(
				[groups_info[0]['zsl'], np.eye(num_total_superitems, dtype=int)], axis=1
			)
			groups_info[0]['ol'] = np.concatenate([
				groups_info[0]['ol'], 
				[superitem_heights[s] for s in range(num_total_superitems)]
			])

	return groups_info, layer_pool


def warm_start_no_groups(superitems, pallet_lenght, pallet_width):
	'''
	Given the whole DataFrame of superitems, return the layer heights
	and superitems to layer assignments based on the maxrects algorithm
	'''
	rects = []
	for _, row in superitems.iterrows():
		rects += [(row.lenght, row.width, row.name)]

	initial_layers = maxrects(rects, pallet_lenght, pallet_width)

	ol = np.zeros((len(initial_layers),), dtype=int)
	zsl = np.zeros((len(superitems), len(initial_layers)), dtype=int)
	for l, layer in enumerate(initial_layers):
		superitems_ids = []
		for rect in layer:
			superitems_ids += [rect[0]]
			zsl[rect[0], l] = 1
		ol[l] = superitems[superitems.index.isin(superitems_ids)].height.max()

	return zsl, ol
