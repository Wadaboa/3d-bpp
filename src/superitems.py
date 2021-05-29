from collections import defaultdict

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

    def __str__(self):
        return (
            f"Dimension(width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Coordinate:
    """
    Helper class to define a triplet of coordinates
    """

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()


class Item:
    """
    An item is a single product
    """

    def __init__(self, width, depth, height, weight):
        self.dimensions = Dimension(width, depth, height, weight)

    @classmethod
    def from_series(cls, item):
        """
        Return an Item from a Pandas Series having the expected columns
        """
        return Item(item.width, item.depth, item.height, item.weight)

    @classmethod
    def from_dataframe(cls, order):
        """
        Return a list of Item objects from a Pandas DataFrame
        having the expected columns
        """
        return [Item(i.width, i.depth, i.height, i.weight) for _, i in order.iterrows()]

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

    def __str__(self):
        return (
            f"Item(width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


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

    def get_item_coords(self, height=0):
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

    def get_items(self):
        """
        Return a list of single items in the superitem
        """
        items = []
        for item in self.items:
            if isinstance(item, Item):
                items.append(item)
                continue
            items += item.get_items()
        return items

    def get_num_items(self):
        """
        Return the number of single items in the superitem
        """
        return len(self.get_items())

    def __str__(self):
        return (
            f"Superitem(width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume}, coords={self.get_item_coords()})"
        )

    def __repr__(self):
        return self.__str__()


class HorizontalSuperitem(Superitem):
    """
    An horizontal superitem is a group of 2 or 4 items (not superitems)
    that have exactly the same dimensions and get stacked next to each other
    """

    def __init__(self, items):
        super().__init__(items)


class TwoHorizontalSuperitemWidth(HorizontalSuperitem):
    """
    Horizontal superitem with 2 items stacked by the width dimension
    """

    def __init__(self, items):
        assert len(items) == 2
        super().__init__(items)

    def _get_superitem(self):
        i1, i2 = tuple(self.items)
        return Item(i1.width * 2, i1.depth, i1.height, i1.weight + i2.weight)

    def get_item_coords(self, height=0):
        i1, i2 = tuple(self.items)
        return [Coordinate(0, 0, height), Coordinate(i1.width, 0, height)]


class TwoHorizontalSuperitemDepth(HorizontalSuperitem):
    """
    Horizontal superitem with 2 items stacked by the depth dimension
    """

    def __init__(self, items):
        assert len(items) == 2
        super().__init__(items)

    def _get_superitem(self):
        i1, i2 = tuple(self.items)
        return Item(i1.width, i1.depth * 2, i1.height, i1.weight + i2.weight)

    def get_item_coords(self, height=0):
        i1, i2 = tuple(self.items)
        return [Coordinate(0, 0, height), Coordinate(0, i1.depth, height)]


class FourHorizontalSuperitem(HorizontalSuperitem):
    """
    Horizontal superitem with 4 items stacked by the width and depth dimensions
    """

    def __init__(self, items):
        assert len(items) == 4
        super().__init__(items)

    def _get_superitem(self):
        i1, i2, i3, i4 = tuple(self.items)
        return Item(
            i1.width * 2, i1.depth * 2, i1.height, i1.weight + i2.weight + i3.weight + i4.weight
        )

    def get_item_coords(self, height=0):
        i1, i2 = tuple(self.items)
        return [
            Coordinate(0, 0, height),
            Coordinate(i1.width, 0, height),
            Coordinate(0, i1.depth, height),
            Coordinate(i1.width, i1.depth, height),
        ]


class VerticalSuperitem(Superitem):
    """
    A vertical superitem is a group of >= 2 items or superitems
    that have similar dimensions and get stacked on top of each other
    """

    def __init__(self, items):
        super().__init__(items)

    def _get_superitem(self):
        return Item(
            max(i.width for i in self.items),
            max(i.depth for i in self.items),
            sum(i.height for i in self.items),
            sum(i.weight for i in self.items),
        )

    def get_item_coords(self, height=0):
        all_coords = []
        for item in self.items:
            if isinstance(item, Item):
                coords = [Coordinate(0, 0, height)]
            elif isinstance(item, HorizontalSuperitem):
                coords = item.get_item_coords(height=height)
            all_coords.append(coords)
            height += item.height
        return all_coords


class SuperitemPool:
    """
    Set of superitems for a given order
    """

    def __init__(self, order, pallet_dims, max_vstacked=2):
        self.items = Item.from_dataframe(order)
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
        superitems_vertical = self._gen_superitems_vertical(self.items + superitems_horizontal)
        superitems = self.items + superitems_horizontal + superitems_vertical
        superitems = self._filter_superitems(superitems)
        return superitems

    def _gen_superitems_horizontal(self):
        """
        Horizontally stack groups of 2 and 4 items with the same
        dimensions to form single superitems
        """
        superitems_horizontal = []

        # Get items having the exact same dimensions
        dims = [(i.width, i.depth, i.height) for i in self.items]
        indexes = list(range(len(dims)))
        same_dims = defaultdict(list)
        for k, v in zip(dims, indexes):
            same_dims[k].append(v)

        # Extract candidate groups made up of 2 and 4 items
        two_slices, four_slices = [], []
        for _, indexes in same_dims.items():
            two_slices += [
                (self.items[indexes[i]], self.items[indexes[i + 1]])
                for i in range(0, len(indexes) - 1, 2)
            ]
            four_slices += [
                (
                    self.items[indexes[i]],
                    self.items[indexes[i + 1]],
                    self.items[indexes[i + 2]],
                    self.items[indexes[i + 3]],
                )
                for i in range(0, len(indexes) - 3, 4)
            ]

        # Generate 2-items horizontal superitems
        for slice in tqdm(two_slices, desc="Generating horizontal 2-items superitems"):
            items = [Item.from_series(p) for p in slice]
            superitems_horizontal += [
                TwoHorizontalSuperitemWidth(items),
                TwoHorizontalSuperitemDepth(items),
            ]

        # Generate 4-items horizontal superitems
        for slice in tqdm(four_slices, desc="Generating horizontal 4-items superitems"):
            items = [Item.from_series(p) for p in slice]
            superitems_horizontal += [FourHorizontalSuperitem(items)]

        return superitems_horizontal

    def _gen_superitems_vertical(self, superitems):
        """
        Vertically stack groups of >= 2 items or superitems with the
        same dimensions to form a taller superitem
        """
        superitems_vertical = []

        # Add the "width * depth" column and sort superitems
        # in ascending order by that dimension
        wd = [s.width * s.depth for s in superitems]
        superitems = [superitems[i] for i in np.argsort(wd)]

        # Extract candidate groups made up of >= 2 items or superitems
        slices = []
        for s in range(2, self.max_vstacked + 1):
            slices += [
                tuple(superitems[i + j] for j in range(s))
                for i in range(0, len(superitems) - (s - 1), s)
            ]

        # Generate vertical superitems
        for slice in slices:
            if slice[0].width * slice[0].depth >= 0.7 * slice[-1].width * slice[-1].depth:
                items = [Item.from_series(p) for p in slice]
                superitems_vertical += [VerticalSuperitem(items)]

        return superitems_vertical

    def _filter_superitems(self, superitems):
        """
        Keep only those superitems that do not exceed the
        pallet capacity
        """
        return [
            s
            for s in superitems
            if s.width <= self.pallet_width
            and s.depth <= self.pallet_depth
            and s.height <= self.pallet_height
        ]

    def get_superitems_dims(self):
        """
        Return the dimensions of superitems in the pool
        as 3 numpy arrays
        """
        ws = [s.width for s in self.superitems]
        ds = [s.depth for s in self.superitems]
        hs = [s.height for s in self.superitems]
        return ws, ds, hs

    def __getitem__(self, i):
        return self.superitems[i]

    def __str__(self):
        return f"SuperitemPool(superitems={self.superitems})"

    def __repr__(self):
        return self.__str__()


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
