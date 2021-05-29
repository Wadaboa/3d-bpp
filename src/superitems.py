import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils


class Dimension:
    """
    Helper class to define object dimensions
    """

    def __init__(self, width, depth, height, weight=None):
        self.width = int(width)
        self.depth = int(depth)
        self.height = int(height)
        self.weight = int(weight)
        self.volume = int(width * depth * height)


class Coordinate:
    """
    Helper class to define a triplet of coordinates
    """

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class Item:
    """
    An item is a single product
    """

    def __init__(self, width, depth, height, weight):
        self.dimensions = Dimension(width, depth, height, weight)

    @property
    def width(self):
        return self.dimensions.width

    @property
    def depth(self):
        return self.dimensions.depth

    @property
    def height(self):
        return self.dimensions.height

    @property
    def weight(self):
        return self.dimensions.weight

    @property
    def volume(self):
        return self.dimensions.volume

    def get_coords(self):
        return Coordinate(0, 0, 0)


class Superitem:
    """
    A superitem is a grouping of items or superitems
    having almost the same dimensions
    """

    def __init__(self, items):
        self.items = items
        self.superitem = self._get_superitem()

    def _get_superitem(self):
        raise NotImplementedError()

    @property
    def width(self):
        return self.superitem.width

    @property
    def depth(self):
        return self.superitem.depth

    @property
    def height(self):
        return self.superitem.height

    @property
    def weight(self):
        return self.superitem.weight

    @property
    def volume(self):
        return self.superitem.volume


class TwoHorizontalSuperitemWidth(Superitem):
    def __init__(self, items):
        assert len(items) == 2
        super.__init__(items=items)

    def _get_superitem(self):
        i1, i2 = tuple(self.items)
        return Item(i1.width * 2, i1.depth, i1.height, i1.weight + i2.weight)


class TwoHorizontalSuperitemDepth(Superitem):
    def __init__(self, items):
        assert len(items) == 2
        super.__init__(items=items)

    def _get_superitem(self):
        i1, i2 = tuple(self.items)
        return Item(i1.width, i1.depth * 2, i1.height, i1.weight + i2.weight)


class FourHorizontalSuperitem(Superitem):
    """
    An horizontal superitem is a group of 2 or 4 items (not superitems)
    that have exactly the same dimensions and get stacked next to each other
    """

    def __init__(self, items):
        assert len(items) == 4
        super.__init__(items=items)

    def _get_superitem(self):
        i1, i2, i3, i4 = tuple(self.items)
        return Item(
            i1.width * 2, i1.depth * 2, i1.height, i1.weight + i2.weight + i3.weight + i4.weight
        )


class VerticalSuperitem(Superitem):
    """
    A vertical superitem is a group of >= 2 items or superitems
    that have similar dimensions and get stacked on top of each other
    """

    def __init__(self, items):
        super.__init__(items=items)

    def _get_superitem(self):
        return Item(
            max(i.width for i in self.items),
            max(i.depth for i in self.items),
            sum(i.height for i in self.items),
            sum(i.weight for i in self.items),
        )


class SuperitemPool:
    """
    Set of superitems for a given order
    """

    def __init__(self, order, pallet_dims, max_vstacked=2):
        self.order = order
        self.pallet_width, self.pallet_depth, self.pallet_height = pallet_dims
        self.max_vstacked = max_vstacked
        self.superitems = self._gen_superitems()

    def _gen_superitems(self):
        """
        Group and stack items both horizontally and vertically
        to form superitems
        """
        # Generate horizontal and vertical superitems and
        # filter the ones exceeding the pallet dimensions
        superitems_horizontal = self._gen_superitems_horizontal()
        superitems_vertical = self._gen_superitems_vertical()
        superitems = pd.concat([self.items, superitems_horizontal, superitems_vertical])
        superitems = self._filter_superitems(superitems)

        # Add useful columns
        superitems.loc[:, "flattened_items"] = superitems["items"].map(
            lambda l: list(utils.flatten(l))
        )
        superitems.loc[:, "num_items"] = superitems["flattened_items"].str.len()

        return superitems

    def _gen_superitems_horizontal(self):
        """
        Horizontally stack groups of 2 and 4 items with the same
        dimensions to form single superitems
        """
        superitems_horizontal = []

        # Get items having the exact same dimensions
        same_dims = (
            self.order.reset_index()
            .groupby(["width", "depth", "height"], as_index=False)
            .agg({"index": list})
        )

        # Extract candidate groups made up of 2 and 4 items
        two_slices, four_slices = [], []
        for _, dims in same_dims.iterrows():
            vals = dims["index"]
            two_slices += [
                (self.order.iloc[vals[i]], self.order.iloc[vals[i + 1]])
                for i in range(0, len(vals) - 1, 2)
            ]
            four_slices += [
                (
                    self.order.iloc[vals[i]],
                    self.order.iloc[vals[i + 1]],
                    self.order.iloc[vals[i + 2]],
                    self.order.iloc[vals[i + 3]],
                )
                for i in range(0, len(vals) - 3, 4)
            ]

        # Generate 2-items horizontal superitems
        for p1, p2 in tqdm(two_slices, desc="Generating horizontal 2-items superitems"):
            superitems_horizontal += [
                [
                    [p1.name, p2.name],
                    p1["width"] * 2,
                    p1["depth"],
                    p1["height"],
                    p1["weight"] + p2["weight"],
                    p1["volume"] + p2["volume"],
                    False,
                ],
                [
                    [p1.name, p2.name],
                    p1["width"],
                    p1["depth"] * 2,
                    p1["height"],
                    p1["weight"] + p2["weight"],
                    p1["volume"] + p2["volume"],
                    False,
                ],
            ]

        # Generate 4-items horizontal superitems
        for p1, p2, p3, p4 in tqdm(four_slices, desc="Generating horizontal 4-items superitems"):
            superitems_horizontal += [
                [
                    [p1.name, p2.name, p3.name, p4.name],
                    p1["width"] * 2,
                    p1["depth"] * 2,
                    p1["height"],
                    p1["weight"] + p2["weight"] + p3["weight"] + p4["weight"],
                    p1["volume"] + p2["volume"] + p3["volume"] + p4["volume"],
                    False,
                ]
            ]

        # Make single items DataFrame have the same schema
        # as the superitems one
        items = self.order.reset_index().drop(columns="id").rename(columns={"index": "items"})
        items["items"] = items["items"].apply(lambda x: [x])
        items["vstacked"] = [False] * len(items)

        # Merge single items and horizontal superitems
        superitems_horizontal = pd.DataFrame(superitems_horizontal, columns=items.columns)
        return pd.concat([items, superitems_horizontal])

    def _gen_superitems_vertical(self, superitems):
        """
        Vertically stack groups of 2 items or superitems with the
        same dimensions to form a taller superitem
        """
        # Add the "width * height" column and sort superitems
        # in ascending order by that dimension
        superitems["wh"] = superitems["width"] * superitems["height"]
        superitems = superitems.sort_values(["wh", "depth"]).reset_index(drop=True)

        # Extract candidate groups made up of 2 items
        slices = []
        for s in range(2, self.max_vstacked + 1):
            slices += [
                tuple(superitems.iloc[i + j] for j in range(s))
                for i in range(0, len(superitems) - (s - 1), s)
            ]

        # Generate 2-items vertical superitems
        for slice in slices:
            if slice[0]["lw"] >= 0.7 * slice[-1]["lw"]:
                superitems = superitems.append(
                    {
                        "items": [i["items"] for i in slice],
                        "width": max(i["width"] for i in slice),
                        "depth": max(i["depth"] for i in slice),
                        "height": sum(i["height"] for i in slice),
                        "weight": sum(i["weight"] for i in slice),
                        "volume": sum(i["volume"] for i in slice),
                        "wh": slice[-1]["wh"],
                        "vstacked": True,
                    },
                    ignore_index=True,
                )

        return superitems.drop(columns="wh")

    def _filter_superitems(self, superitems):
        """
        Keep only those superitems that do not exceed the
        pallet capacity
        """
        return superitems[
            (superitems.width <= self.pallet_width)
            & (superitems.height <= self.pallet_height)
            & (superitems.depth <= self.pallet_depth)
        ].reset_index(drop=True)

    def get_superitems_dims(self):
        """
        Return the dimensions of superitems in the pool
        as 3 numpy arrays
        """
        ws = self.superitems.width.values
        ds = self.superitems.height.values
        hs = self.superitems.depth.values
        return ws, ds, hs


def select_superitems_group(superitems, ids):
    keys = np.array(list(ids.keys()), dtype=int)
    sub_superitems = superitems.iloc[keys]
    sub_ws = sub_superitems.width.values
    sub_ds = sub_superitems.height.values
    sub_hs = sub_superitems.depth.values
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
