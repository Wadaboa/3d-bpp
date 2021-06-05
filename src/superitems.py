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

    def __init__(self, id, width, depth, height, weight):
        self.id = id
        self.dimensions = Dimension(width, depth, height, weight)

    @classmethod
    def from_series(cls, item):
        """
        Return an Item from a Pandas Series having the expected columns
        """
        return Item(item.name, item.width, item.depth, item.height, item.weight)

    @classmethod
    def from_dataframe(cls, order):
        """
        Return a list of Item objects from a Pandas DataFrame
        having the expected columns
        """
        return [Item(i.name, i.width, i.depth, i.height, i.weight) for _, i in order.iterrows()]

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
            f"Item(id={self.id}, width={self.width}, depth={self.depth}, "
            f"height={self.height}, weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Superitem:
    """
    A superitem is a grouping of items or superitems
    having almost the same dimensions
    """

    def __init__(self, items):
        # Represents a list of superitems
        self.items = items

    def get_item_coords(self, height=0):
        raise NotImplementedError()

    @property
    def width(self):
        raise NotImplementedError()

    @property
    def depth(self):
        raise NotImplementedError()

    @property
    def height(self):
        raise NotImplementedError()

    @property
    def weight(self):
        return sum(i.weight for i in self.items)

    @property
    def volume(self):
        return int(self.width * self.depth * self.height)

    @property
    def id(self):
        return list(utils.flatten([i.id for i in self.items]))

    def get_items(self):
        """
        Return a list of raw items in the superitem
        """
        return list(utils.flatten([i.items for i in self.items]))

    def get_num_items(self):
        """
        Return the number of single items in the superitem
        """
        return len(self.id)

    def __str__(self):
        return (
            f"Superitem(ids={self.id}, width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume}, coords={self.get_item_coords()})"
        )

    def __repr__(self):
        return self.__str__()


class SingleItemSuperitem(Superitem):
    """
    Superitem containing a single item
    """

    def __init__(self, items):
        assert len(items) == 1
        super().__init__(items)

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

    @property
    def height(self):
        return max(i.height for i in self.items)

    def get_item_coords(self, height=0):
        return [Coordinate(0, 0, height)]


class HorizontalSuperitem(Superitem):
    """
    An horizontal superitem is a group of 2 or 4 items (not superitems)
    that have exactly the same dimensions and get stacked next to each other
    """

    def __init__(self, items):
        super().__init__(items)

    @property
    def height(self):
        return max(i.height for i in self.items)


class TwoHorizontalSuperitemWidth(HorizontalSuperitem):
    """
    Horizontal superitem with 2 items stacked by the width dimension
    """

    def __init__(self, items):
        assert len(items) == 2
        super().__init__(items)

    @property
    def width(self):
        return sum(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

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

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return sum(i.depth for i in self.items)

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

    @property
    def width(self):
        return sum(i.width for i in self.items)

    @property
    def depth(self):
        return sum(i.depth for i in self.items)

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
    A vertical superitem is a group of >= 2 items or horizontal superitems
    that have similar dimensions and get stacked on top of each other
    """

    def __init__(self, items):
        super().__init__(items)

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

    @property
    def height(self):
        return sum(i.height for i in self.items)

    def get_item_coords(self, height=0):
        all_coords = []
        for item in self.items:
            all_coords += item.get_item_coords(height=height)
            height += item.height
        return all_coords


class SuperitemPool:
    """
    Set of superitems for a given order
    """

    def __init__(self, order=None, pallet_dims=None, max_vstacked=2, superitems=None):
        assert superitems is not None or (superitems is None and order is not None)
        self.superitems = (
            self._gen_superitems(order, pallet_dims, max_vstacked)
            if superitems is None
            else superitems
        )
        self.fsi, self.from_index_to_item_id, self.from_item_id_to_index = self._get_fsi()

    def _gen_superitems(self, order, pallet_dims, max_vstacked):
        """
        Group and stack items both horizontally and vertically
        to form superitems
        """
        # Generate horizontal and vertical superitems and
        # filter the ones exceeding the pallet dimensions
        items = Item.from_dataframe(order)
        single_items_superitems = self._gen_single_items_superitems(items)
        superitems_horizontal = self._gen_superitems_horizontal(single_items_superitems)
        superitems_vertical = self._gen_superitems_vertical(
            single_items_superitems + superitems_horizontal, max_vstacked
        )
        superitems = single_items_superitems + superitems_horizontal + superitems_vertical
        if pallet_dims is not None:
            superitems = self._filter_superitems(superitems, pallet_dims)
        return superitems

    def _gen_single_items_superitems(self, items):
        """
        Generate superitems with a single item
        """
        return [SingleItemSuperitem([i]) for i in items]

    def _gen_superitems_horizontal(self, items):
        """
        Horizontally stack groups of 2 and 4 items with the same
        dimensions to form single superitems
        """
        superitems_horizontal = []

        # Get items having the exact same dimensions
        dims = [(i.width, i.depth, i.height) for i in items]
        indexes = list(range(len(dims)))
        same_dims = defaultdict(list)
        for k, v in zip(dims, indexes):
            same_dims[k].append(v)

        # Extract candidate groups made up of 2 and 4 items
        two_slices, four_slices = [], []
        for _, indexes in same_dims.items():
            two_slices += [
                (items[indexes[i]], items[indexes[i + 1]]) for i in range(0, len(indexes) - 1, 2)
            ]
            four_slices += [
                (
                    items[indexes[i]],
                    items[indexes[i + 1]],
                    items[indexes[i + 2]],
                    items[indexes[i + 3]],
                )
                for i in range(0, len(indexes) - 3, 4)
            ]

        # Generate 2-items horizontal superitems
        for slice in tqdm(two_slices, desc="Generating horizontal 2-items superitems"):
            superitems_horizontal += [
                TwoHorizontalSuperitemWidth(slice),
                TwoHorizontalSuperitemDepth(slice),
            ]

        # Generate 4-items horizontal superitems
        for slice in tqdm(four_slices, desc="Generating horizontal 4-items superitems"):
            superitems_horizontal += [FourHorizontalSuperitem(slice)]

        return superitems_horizontal

    def _gen_superitems_vertical(self, superitems, max_vstacked):
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
        for s in range(2, max_vstacked + 1):
            slices += [
                tuple(superitems[i + j] for j in range(s))
                for i in range(0, len(superitems) - (s - 1), s)
            ]

        # Generate vertical superitems
        for slice in slices:
            if slice[0].width * slice[0].depth >= 0.7 * slice[-1].width * slice[-1].depth:
                superitems_vertical += [VerticalSuperitem(slice)]

        return superitems_vertical

    def _filter_superitems(self, superitems, pallet_dims):
        """
        Keep only those superitems that do not exceed the
        pallet capacity
        """
        pallet_width, pallet_depth, pallet_height = pallet_dims
        return [
            s
            for s in superitems
            if s.width <= pallet_width and s.depth <= pallet_depth and s.height <= pallet_height
        ]

    def _get_fsi(self):
        """
        Return a binary matrix of superitems by items, s.t.
        fsi[s, i] = 1 iff superitems s contains item i
        """
        item_ids = sorted(self.get_unique_item_ids())
        indexes = list(range(len(item_ids)))
        from_index_to_item_id = dict(zip(item_ids, indexes))
        from_item_id_to_index = dict(zip(item_ids, indexes))

        fsi = np.zeros((len(self.superitems), self.get_num_unique_items()), dtype=int)
        for s, superitem in enumerate(self.superitems):
            for item_id in superitem.id:
                fsi[s, from_item_id_to_index[item_id]] = 1

        return fsi, from_index_to_item_id, from_item_id_to_index

    def get_superitems_dims(self):
        """
        Return the dimensions of superitems in the pool
        as 3 numpy arrays
        """
        ws = [s.width for s in self.superitems]
        ds = [s.depth for s in self.superitems]
        hs = [s.height for s in self.superitems]
        return ws, ds, hs

    def get_item_ids(self):
        """
        Return the ids of items inside the superitem pool
        """
        return [s.id for s in self.superitems]

    def get_unique_item_ids(self):
        """
        Return the flattened list of item ids inside the superitem pool
        """
        return list(set(utils.flatten(self.get_item_ids())))

    def get_num_unique_items(self):
        """
        Return the total number of unique items inside the superitem pool
        """
        return len(self.get_unique_item_ids())

    def to_dataframe(self):
        """
        Convert the superitem pool to a DataFrame instance
        """
        ws, ds, hs = self.get_superitems_dims()
        ids = self.get_item_ids()
        return pd.DataFrame({"width": ws, "depth": ds, "height": hs, "ids": ids})

    def __getitem__(self, i):
        return self.superitems[i]

    def __str__(self):
        return f"SuperitemPool(superitems={self.superitems})"

    def __repr__(self):
        return self.__str__()
