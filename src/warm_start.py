import numpy as np
from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf

from . import utils


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
        group.loc[:, "flattened_items"] = group["items"].map(lambda l: list(utils.flatten(l)))
        group.loc[:, "num_items"] = group["flattened_items"].str.len()
        sorted_group = group.sort_values(
            by=["num_items", "stacked"], ascending=False
        ).reset_index().rename(columns={"index": "superitem_id"})
        indexes_to_remove = []
        for i, row in sorted_group.iterrows():
            for item in row["flattened_items"]:
                to_remove = sorted_group[
                    (sorted_group["flattened_items"].map(lambda l: item in l)) &
                    (sorted_group.index > i)
                ]
                indexes_to_remove += to_remove.index.tolist()

        final_groups += [sorted_group.drop(index=indexes_to_remove)]

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


def warm_start(num_superitems, groups, pallet_lenght, pallet_width):
	ol, zsl = [], []
	for group in groups:
		rects = []
		for _, row in group.iterrows():
			rects += [(row.lenght, row.width, row.superitem_id)]

		layers = maxrects(rects, pallet_lenght, pallet_width)

		superitems_in_layer = np.full((num_superitems, len(layers)), False)
		for l, layer in enumerate(layers):
			superitems_ids = []
			for rect in layer:
				superitems_ids += [rect[0]]
				superitems_in_layer[rect[0], l] = True
			ol += [group[group.superitem_id.isin(superitems_ids)].height.max()]

		zsl += [superitems_in_layer]
	
	zsl = np.concatenate(zsl, axis=1).astype(int)
	return zsl, ol
