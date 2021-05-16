import itertools
from typing import final

import numpy as np
import pandas as pd

from . import utils


def generate_superitems(order, pallet_dims):
    superitems_horizontal, superitems_vertical = [], []
    same_dims = order.reset_index().groupby(
        ['width', 'lenght', 'height'], as_index=False
    ).agg({'index': list, 'id': list})
    for _, row in same_dims.iterrows():
        if len(row["index"]) > 1:
            for i1, i2 in itertools.combinations(row["index"], 2):
                p1, p2 = order.iloc[i1], order.iloc[i2]
                superitems_horizontal += [
                    [row["index"], row["id"], p1["lenght"] * 2, p1["width"], p1["height"], p1["weight"] + p2["weight"], False],
                    [row["index"], row["id"], p1["lenght"], p1["width"] * 2, p1["height"], p1["weight"] + p2["weight"], False]
                ]
        if len(row["index"]) > 3:
            for i1, i2, i3, i4 in itertools.combinations(row["index"], 4):
                p1, p2, p3, p4 = order.iloc[i1], order.iloc[i2], order.iloc[i3], order.iloc[i4]
                superitems_horizontal += [[
                    row["index"],
                    row["id"],
                    p1["lenght"] * 2, 
                    p1["width"] * 2, 
                    p1["height"],
                    p1["weight"] + p2["weight"] + p3["weight"] + p4["weight"],
                    False
                ]]
    
    items = order.reset_index().rename(columns={"index": "items"})
    items["id"] = items["id"].apply(lambda x: [x])
    items["items"] = items["items"].apply(lambda x: [x])
    items["stacked"] = [False] * len(items)
    superitems_horizontal = pd.DataFrame(superitems_horizontal, columns=items.columns)
    items_superitems = pd.concat([items, superitems_horizontal]).reset_index(drop=True).sort_values(
        ["width", "lenght", "height"]
    )

    for a, i in items_superitems.iterrows():
        for b, j in items_superitems.iterrows():
            if (b > a and (len(set(i["items"]).intersection(set(j["items"]))) == 0) and 
                (j["width"] * j["lenght"] >= i["width"] * i["lenght"]) and
                (i["width"] * i["lenght"] >= 0.7 * j["width"] * j["lenght"])):
                    superitems_vertical += [[
                        [i["items"], j["items"]],
                        [i["id"], j["id"]],
                        max(i["lenght"], j["lenght"]), 
                        max(i["width"], j["width"]), 
                        i["height"] + j["height"],
                        i["weight"] + j["weight"],
                        True
                    ]]
                    
    superitems_vertical = pd.DataFrame(superitems_vertical, columns=items.columns)
    final_superitems = pd.concat([items_superitems, superitems_vertical])
    
    pallet_lenght, pallet_width, pallet_height = pallet_dims
    final_superitems = final_superitems[
        (final_superitems.lenght <= pallet_lenght) &
        (final_superitems.width <= pallet_width) &
        (final_superitems.height <= pallet_height)
    ].reset_index(drop=True)
    
    ws = final_superitems.lenght.values
    ds = final_superitems.width.values
    hs = final_superitems.height.values

    return final_superitems, ws, ds, hs

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
    print(item_ids)
    sub_fsi = np.zeros((len(keys), len(sub_items)), dtype=int)
    for i in item_ids:
        for s in keys:
            if fsi[s, i] == 1:
                sub_fsi[ids[s], item_ids[i]] = 1
    return sub_fsi
