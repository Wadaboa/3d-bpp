import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils


class Layer():

    def __init__(self, height, superitem_ids, coords):
        self.height = height
        self.superitem_ids = superitem_ids
        self.coords = coords

    def get_superitems(self, superitems):
        return superitems.iloc[self.superitem_ids]

    def get_item_ids(self, superitems):
        return list(utils.flatten(self.get_superitems(superitems).flattened_items.to_list()))
    
    def get_items(self, order, superitems):
        item_ids = self.get_item_ids(superitems)
        return order.iloc[item_ids]

    def get_volume(self, superitems):
        return sum(superitems.iloc[s].volume for s in self.superitem_ids)

    def get_density(self, superitems, W, D):
        return self.get_volume(superitems) / W * D * self.height

    def map_superitem_ids(self, mapping):
        self.superitem_ids = [mapping[s] for s in self.superitem_ids]

    def __str__(self):
        return f"Layer(height={self.height}, ids={self.superitem_ids})"

    def __repr__(self):
        return self.__str__()

    
class LayerPool():

    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        assert isinstance(layer, Layer), (
            "The given layer should be an instance of the Layer class"
        )
        self.layers.append(layer)

    def add_pool(self, layer_pool):
        assert isinstance(layer_pool, LayerPool), (
            "The given set of layers should be an instance of the LayerPool class"
        )
        self.layers.extend(layer_pool.layers)

    def map_superitem_ids(self, mapping):
        for layer in self.layers:
            layer.map_superitem_ids(mapping)

    def is_present(self, layer):
        assert isinstance(layer, Layer), (
            "The given layer should be an instance of the Layer class"
        )
        present = False
        for other_layer in self.layers:
            if (utils.np_are_equal(layer.superitem_ids, other_layer.superitem_ids) and 
                    utils.np_are_equal(layer.coords, other_layer.coords)):
                present = True
                break
        return present

    def get_item_ids(self, superitems):
        item_ids = []
        for layer in self.layers:
            item_ids += layer.get_item_ids(superitems)
        return item_ids

    def get_densities(self, superitems, W, D):
        return [layer.get_density(superitems=superitems, W=W, D=D) for layer in self.layers]

    def select_layers(self, superitems, W, D, min_density=0.5):
        # Sort layers by densities and keep only those with a 
        # density greater than or equal to the given minimum
        densities = self.get_densities(superitems, W, D)
        sorted_layers = sorted(self.layers, key=densities, reverse=True)
        sorted_layers = LayerPool(
            layers=[l for d, l in zip(densities, sorted_layers) if d >= min_density]
        )
        
        # Discard layers after all items are covered 
        all_item_ids = sorted_layers.get_item_ids(superitems)
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        selected_layers = LayerPool()
        for layer in sorted_layers:
            # Stop when all items are covered
            if all(list(item_coverage.values())):
                break
            
            # Update coverage
            item_ids = layer.get_item_ids(superitems)
            for item in item_ids:
                item_coverage[item] = True

            # Add the current layer to the pool
            # of selected layers
            selected_layers.add(layer)

        return selected_layers

    def replace_items(self, superitems, max_item_coverage=3):
        all_item_ids = self.get_item_ids(superitems)
        item_coverage = dict(zip(all_item_ids, [0] * len(all_item_ids)))
        selected_layers = LayerPool()
        for layer in self.layers:
            to_select = True
            item_ids = layer.get_item_ids(superitems)
            
            # If at least one item in the layer was already selected
            # more times than the maximum allowed value, then such layer
            # is to be discarded
            for item in item_ids:
                if item_coverage[item] >= max_item_coverage:
                    to_select = False
            
            # If the layer is selected, increase item coverage
            # for each item in such layer and add it to the pool
            # of selected layers
            if to_select:
                selected_layers.add(layer)
                for item in item_ids:
                    item_coverage[item] += 1

            ################################ TODO
            # We also let each selected layer to have a maximum of 3
            # items that are covered using the previously selected layers
            # ##############################
        
        return selected_layers


    def __str__(self):
        return f"LayerPool(layers={self.layers})"
    
    def __repr__(self):
        return self.__str__()

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, e):
        assert isinstance(e, Layer), (
            "The given layer should be an instance of the Layer class"
        )
        self.layers[i] = e



def generate_superitems(order, pallet_dims, max_stacked_items=2):
    superitems_horizontal = []
    same_dims = order.reset_index().groupby(
        ['width', 'lenght', 'height'], as_index=False
    ).agg({'index': list})

    two_slices, four_slices = [], []
    for _, dims in same_dims.iterrows():
        vals = dims["index"]
        two_slices += [(order.iloc[vals[i]], order.iloc[vals[i + 1]]) for i in range(0, len(vals) - 1, 2)]
        four_slices += [(order.iloc[vals[i]], order.iloc[vals[i + 1]], order.iloc[vals[i + 2]], order.iloc[vals[i + 3]]) for i in range(0, len(vals) - 3, 4)]
    
    for p1, p2 in tqdm(two_slices, desc="Generating horizontal 2-items superitems"):
        indexes = [p1.name, p2.name]
        superitems_horizontal += [
            [indexes, p1["lenght"] * 2, p1["width"], p1["height"], p1["weight"] + p2["weight"], p1["volume"] + p2["volume"], False],
            [indexes, p1["lenght"], p1["width"] * 2, p1["height"], p1["weight"] + p2["weight"], p1["volume"] + p2["volume"], False]
        ]

    for p1, p2, p3, p4 in tqdm(four_slices, desc="Generating horizontal 4-items superitems"):
        superitems_horizontal += [[
            [p1.name, p2.name, p3.name, p4.name],
            p1["lenght"] * 2, 
            p1["width"] * 2, 
            p1["height"],
            p1["weight"] + p2["weight"] + p3["weight"] + p4["weight"],
            p1["volume"] + p2["volume"] + p3["volume"] + p4["volume"],
            False
        ]]
    
    items = order.reset_index().drop(columns="id").rename(columns={"index": "items"})
    items["items"] = items["items"].apply(lambda x: [x])
    items["vstacked"] = [False] * len(items)
    superitems_horizontal = pd.DataFrame(superitems_horizontal, columns=items.columns)
    items_superitems = pd.concat([items, superitems_horizontal])
    items_superitems["lw"] = items_superitems["width"] * items_superitems["lenght"]
    items_superitems = items_superitems.sort_values(["lw", "height"]).reset_index(drop=True)

    slices = []
    for s in range(2, max_stacked_items + 1):
        slices += [
            tuple(items_superitems.iloc[i + j] for j in range(s)) 
            for i in range(0, len(items_superitems) - (s - 1), s)
        ]
    
    for slice in slices:
        if slice[0]["lw"] >= 0.7 * slice[-1]["lw"]:
            items_superitems = items_superitems.append({
                "items": [i["items"] for i in slice],
                "lenght": max(i["lenght"] for i in slice), 
                "width": max(i["width"] for i in slice), 
                "height": sum(i["height"] for i in slice),
                "weight": sum(i["weight"] for i in slice),
                "volume": sum(i["volume"] for i in slice),
                "lw": slice[-1]["lw"],
                "vstacked": True
            }, ignore_index=True)

    items_superitems = items_superitems.drop(columns="lw")
    pallet_lenght, pallet_width, pallet_height = pallet_dims
    items_superitems = items_superitems[
        (items_superitems.lenght <= pallet_lenght) &
        (items_superitems.width <= pallet_width) &
        (items_superitems.height <= pallet_height)
    ].reset_index(drop=True)

    items_superitems.loc[:, "flattened_items"] = items_superitems["items"].map(lambda l: list(utils.flatten(l)))
    items_superitems.loc[:, "num_items"] = items_superitems["flattened_items"].str.len()
    
    ws = items_superitems.lenght.values
    ds = items_superitems.width.values
    hs = items_superitems.height.values

    return items_superitems, ws, ds, hs

def select_superitems_group(superitems, ids):
    keys = np.array(list(ids.keys()), dtype=int)
    sub_superitems = superitems.iloc[keys]
    sub_ws = sub_superitems.lenght.values
    sub_ds = sub_superitems.width.values
    sub_hs = sub_superitems.height.values
    return sub_superitems, sub_ws, sub_ds, sub_hs


def items_assignment(superitems):
    n_superitems = len(superitems)
    n_items = len(superitems[superitems["items"].str.len() == 1])
    fsi = np.zeros((n_superitems, n_items), dtype=int)
    for s in range(n_superitems):
        for i in utils.flatten(superitems.loc[s, "items"]):
            fsi[s, i] = 1
    return fsi

def select_fsi_group(fsi, ids):
    keys = np.array(list(ids.keys()), dtype=int)
    sub_items = np.nonzero(fsi[keys])[1].tolist()
    item_ids = dict(zip(sub_items, range(len(sub_items))))
    sub_fsi = np.zeros((len(keys), len(sub_items)), dtype=int)
    for i in item_ids:
        for s in keys:
            if fsi[s, i] == 1:
                sub_fsi[ids[s], item_ids[i]] = 1
    return sub_fsi, item_ids
