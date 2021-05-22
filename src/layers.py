import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils


def generate_superitems(order, pallet_dims, max_stacked_items=4):
    superitems_horizontal = []
    same_dims = order.reset_index().groupby(
        ['width', 'lenght', 'height'], as_index=False
    ).agg({'index': list, 'id': list})

    two_slices, four_slices = [], []
    for _, dims in same_dims.iterrows():
        vals = dims["index"]
        two_slices += [(order.iloc[vals[i]], order.iloc[vals[i + 1]]) for i in range(0, len(vals) - 1, 2)]
        four_slices += [(order.iloc[vals[i]], order.iloc[vals[i + 1]], order.iloc[vals[i + 2]], order.iloc[vals[i + 3]]) for i in range(0, len(vals) - 3, 4)]
    
    for p1, p2 in tqdm(two_slices, desc="Generating horizontal 2-items superitems"):
        indexes = [p1.name, p2.name]
        ids = [p1["id"], p2["id"]]
        superitems_horizontal += [
            [indexes, ids, p1["lenght"] * 2, p1["width"], p1["height"], p1["weight"] + p2["weight"], p1["volume"] + p2["volume"], False],
            [indexes, ids, p1["lenght"], p1["width"] * 2, p1["height"], p1["weight"] + p2["weight"], p1["volume"] + p2["volume"], False]
        ]

    for p1, p2, p3, p4 in tqdm(four_slices, desc="Generating horizontal 4-items superitems"):
        superitems_horizontal += [[
            [p1.name, p2.name, p3.name, p4.name],
            [p1["id"], p2["id"], p3["id"], p4["id"]],
            p1["lenght"] * 2, 
            p1["width"] * 2, 
            p1["height"],
            p1["weight"] + p2["weight"] + p3["weight"] + p4["weight"],
            p1["volume"] + p2["volume"] + p3["volume"] + p4["volume"],
            False
        ]]
    
    items = order.reset_index().rename(columns={"index": "items"})
    items["id"] = items["id"].apply(lambda x: [x])
    items["items"] = items["items"].apply(lambda x: [x])
    items["stacked"] = [False] * len(items)
    superitems_horizontal = pd.DataFrame(superitems_horizontal, columns=items.columns)
    items_superitems = pd.concat([items, superitems_horizontal]).sort_values(
        ["width", "lenght", "height"]
    ).reset_index(drop=True)

    for _ in range(max_stacked_items):
        superitems_vertical = []
        for a, i in tqdm(items_superitems.iterrows(), desc="Generating vertical superitems"):
            for _, j in items_superitems.iloc[a + 1:].iterrows():
                if ((len(set(utils.flatten(i["items"])).intersection(set(utils.flatten(j["items"])))) == 0) and 
                    (j["width"] * j["lenght"] >= i["width"] * i["lenght"]) and
                    (i["width"] * i["lenght"] >= 0.7 * j["width"] * j["lenght"])):
                        superitems_vertical += [[
                            [i["items"], j["items"]],
                            [i["id"], j["id"]],
                            max(i["lenght"], j["lenght"]), 
                            max(i["width"], j["width"]), 
                            i["height"] + j["height"],
                            i["weight"] + j["weight"],
                            i["volume"] + j["volume"],
                            True
                        ]]
                        break
        
        items_superitems = items_superitems.append(pd.DataFrame(superitems_vertical, columns=items.columns), ignore_index=True)

    pallet_lenght, pallet_width, pallet_height = pallet_dims
    items_superitems = items_superitems[
        (items_superitems.lenght <= pallet_lenght) &
        (items_superitems.width <= pallet_width) &
        (items_superitems.height <= pallet_height)
    ].reset_index(drop=True)
    
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
    return sub_fsi
