from collections import defaultdict
from loguru import logger

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils


class Item:
    """
    An item is a single product
    """

    def __init__(self, id, width, depth, height, weight):
        self.id = id
        self.dimensions = utils.Dimension(width, depth, height, weight)

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

    @property
    def area(self):
        return self.dimensions.area

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id and self.dimensions == other.dimensions
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def get_items_coords(self, width=0, depth=0, height=0):
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
    def enclosing_volume(self):
        raise NotImplementedError()

    @property
    def weight(self):
        return sum(i.weight for i in self.items)

    @property
    def volume(self):
        return sum(i.volume for i in self.items)

    @property
    def area(self):
        return sum(i.area for i in self.items)

    @property
    def id(self):
        return sorted(utils.flatten([i.id for i in self.items]))

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

    def get_items_dims(self):
        all_dims = dict()
        for i in range(len(self.items)):
            dims = self.items[i].get_items_dims()
            dups = utils.duplicate_keys([all_dims, dims])
            assert len(dups) == 0, f"Duplicated item in the same superitem, Items id:{dups}"
            all_dims = {**all_dims, **dims}
        return all_dims

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.id == other.id
                and self.width == other.width
                and self.depth == other.depth
                and self.height == other.height
                and self.weight == other.weight
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Superitem(ids={self.id}, width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume}, coords={self.get_items_coords()})"
        )

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum(hash(str(i)) for i in self.id)


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

    @property
    def enclosing_volume(self):
        return self.volume

    def get_items_coords(self, width=0, depth=0, height=0):
        return {self.items[0].id: utils.Coordinate(width, depth, height)}

    def get_items_dims(self):
        return {self.items[0].id: self.items[0].dimensions}


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

    @property
    def enclosing_volume(self):
        return self.volume


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

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=width + i1.width, depth=depth, height=height)
        dups = utils.duplicate_keys([d1, d2])
        assert len(dups) == 0, f"Duplicated item in the same superitem, Items id:{dups}"
        return {**d1, **d2}


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

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=width, depth=i1.depth + depth, height=height)
        dups = utils.duplicate_keys([d1, d2])
        assert len(dups) == 0, f"Duplicated item in the same superitem, Items id:{dups}"
        return {**d1, **d2}


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

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2, i3, i4 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=i1.width + width, depth=depth, height=height)
        d3 = i3.get_items_coords(width=width, depth=i1.depth + depth, height=height)
        d4 = i4.get_items_coords(width=i1.width + width, depth=i1.depth + depth, height=height)
        dups = utils.duplicate_keys([d1, d2, d3, d4])
        assert len(dups) == 0, f"Duplicated item in the same superitem, Items id:{dups}"
        return {**d1, **d2, **d3, **d4}


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

    @property
    def area(self):
        return self.width * self.depth

    @property
    def enclosing_volume(self):
        return self.width * self.depth * self.height

    def get_items_coords(self, width=0, depth=0, height=0):
        # Adjust coordinates to account for stacking tolerance
        all_coords = dict()
        for i in range(len(self.items)):
            width_offset = ((self.width - self.items[i].width) // 2) + width
            depth_offset = ((self.depth - self.items[i].depth) // 2) + depth
            coords = self.items[i].get_items_coords(
                width=width_offset,
                depth=depth_offset,
                height=height,
            )
            dups = utils.duplicate_keys([all_coords, coords])
            assert len(dups) == 0, f"Duplicated item in the same superitem, Items id:{dups}"
            all_coords = {**all_coords, **coords}
            height += self.items[i].height

        return all_coords


class SuperitemPool:
    """
    Set of superitems for a given order
    """

    def __init__(self, superitems=None):
        self.superitems = superitems or []
        self.hash_to_index = self._get_hash_to_index()

    def _get_hash_to_index(self):
        """
        Compute a mapping for all superitems in the SuperitemPool
        with key the hash of the Superitem and value its index in the SuperitemPool
        """
        return {hash(s): i for i, s in enumerate(self.superitems)}

    def subset(self, superitems_indices):
        """
        Return a new superitems pool with the given subset of superitems
        """
        superitems = [s for i, s in enumerate(self.superitems) if i in superitems_indices]
        return SuperitemPool(superitems=superitems)

    def difference(self, superitems_indices):
        """
        Return a new superitems pool without the given subset of superitems
        """
        superitems = [s for i, s in enumerate(self.superitems) if i not in superitems_indices]
        return SuperitemPool(superitems=superitems)

    def add(self, superitem):
        """
        Add the given Superitem to the current SuperitemPool
        """
        assert isinstance(
            superitem, Superitem
        ), "The given Superitem should be an instance of the Superitem class"
        s_hash = hash(superitem)
        if s_hash not in self.hash_to_index:
            self.superitems.append(superitem)
            self.hash_to_index[s_hash] = len(self.superitems) - 1

    def extend(self, superitems_pool):
        """
        Extend the current SuperitemPool with the given one
        """
        assert isinstance(superitems_pool, SuperitemPool) or isinstance(
            superitems_pool, list
        ), "The given set of superitems should be an instance of the SuperitemPool class or a list"
        for superitem in superitems_pool:
            self.add(superitem)

    def remove(self, superitem):
        """
        Remove the given Superitem from the SuperitemPool
        """
        assert isinstance(
            superitem, Superitem
        ), "The given superitem should be an instance of the Superitem class"
        s_hash = hash(superitem)
        if s_hash in self.hash_to_index:
            del self.superitems[self.hash_to_index[s_hash]]
            self.hash_to_index = self._get_hash_to_index()

    def pop(self, i):
        """
        Remove the superitem at the given index from the pool
        """
        self.remove(self.superitems[i])

    def get_fsi(self):
        """
        Return a binary matrix of superitems by items, s.t.
        fsi[s, i] = 1 iff superitems s contains item i
        """
        item_ids = sorted(self.get_unique_item_ids())
        indexes = list(range(len(item_ids)))
        from_index_to_item_id = dict(zip(indexes, item_ids))
        from_item_id_to_index = dict(zip(item_ids, indexes))

        fsi = np.zeros((len(self.superitems), self.get_num_unique_items()), dtype=np.int32)
        for s, superitem in enumerate(self):
            for item_id in superitem.id:
                fsi[s, from_item_id_to_index[item_id]] = 1

        return fsi, from_index_to_item_id, from_item_id_to_index

    def get_superitems_dims(self):
        """
        Return the dimensions of each Superitem in the SuperitemPool
        """
        ws = [s.width for s in self.superitems]
        ds = [s.depth for s in self.superitems]
        hs = [s.height for s in self.superitems]
        return ws, ds, hs

    def get_superitems_containing_item(self, item_id):
        """
        Return a list of Superitem containing the given Item id
        """
        superitems, indices = [], []
        for i, superitem in enumerate(self.superitems):
            if item_id in superitem.id:
                superitems += [superitem]
                indices += [i]
        return superitems, indices

    def get_single_superitems(self):
        """
        Return the list of SingleItemSuperitem in the SuperitemPool
        """
        singles = []
        for superitem in self.superitems:
            if isinstance(superitem, SingleItemSuperitem):
                singles += [superitem]
        return singles

    def get_extreme_superitem(self, minimum=False, two_dims=False):
        """
        Return the Superitem with minimum (or maximum) area (or volume)
        in the SuperitemPool and its index
        """
        func = np.argmax if not minimum else np.argmin
        index = (
            func([s.area for s in self.superitems])
            if two_dims
            else func([s.volume for s in self.superitems])
        )
        return self.superitems[index], index

    def get_item_ids(self):
        """
        Return the ids of each Superitem inside the SuperitemPool,
        where each Superitem's id is a list containing the Item ids of which its compose of
        """
        return [s.id for s in self.superitems]

    def get_unique_item_ids(self):
        """
        Return the flattened list of ids of each Item inside the SuperitemPool
        """
        return sorted(set(utils.flatten(self.get_item_ids())))

    def get_num_unique_items(self):
        """
        Return the total number of unique items inside the SuperitemPool
        """
        return len(self.get_unique_item_ids())

    def get_volume(self):
        """
        Return the sum of the volume of the superitems in the SuperitemPool
        """
        return sum(s.volume for s in self.superitems)

    def get_max_height(self):
        """
        Return the maximum height of the superitems in the SuperitemPool
        """
        if len(self.superitems) == 0:
            return 0
        return max(s.height for s in self.superitems)

    def get_index(self, superitem):
        """
        Given a Superitem instance return the index of the Superitem in the SuperitemPool if present,
        else return None
        """
        return self.hash_to_index.get(hash(superitem))

    def to_dataframe(self):
        """
        Convert the SuperitemPool to a DataFrame instance
        """
        ws, ds, hs = self.get_superitems_dims()
        ids = self.get_item_ids()
        types = [s.__class__.__name__ for s in self.superitems]
        return pd.DataFrame({"width": ws, "depth": ds, "height": hs, "ids": ids, "type": types})

    def __len__(self):
        return len(self.superitems)

    def __contains__(self, superitem):
        return hash(superitem) in self.hash_to_index

    def __getitem__(self, i):
        return self.superitems[i]

    def __str__(self):
        return f"SuperitemPool(superitems={self.superitems})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def gen_superitems(
        cls,
        order,
        pallet_dims,
        max_vstacked=2,
        only_single=False,
        horizontal=True,
        horizontal_type="two-width",
    ):
        """
        Generate horizontal and vertical superitems and
        filter the ones exceeding the pallet dimensions
        """
        logger.info("Generating Superitems")
        items = Item.from_dataframe(order)
        superitems = cls._gen_single_items_superitems(items)
        if only_single:
            return superitems
        if horizontal:
            superitems += cls._gen_superitems_horizontal(superitems, htype=horizontal_type)
            superitems = cls._drop_singles_in_horizontal(superitems)
        superitems += cls._gen_superitems_vertical(superitems, max_vstacked)
        superitems = cls._filter_superitems(superitems, pallet_dims)
        logger.info(f"Generated {len(superitems)} Superitems")
        return superitems

    @classmethod
    def _gen_single_items_superitems(cls, items):
        """
        Generate superitems with a single item
        """
        si_superitems = [SingleItemSuperitem([i]) for i in items]
        logger.debug(f"Generated {len(si_superitems)} SingleItemSuperitems")
        return si_superitems

    @classmethod
    def _gen_superitems_horizontal(cls, items, htype="two-width"):
        """
        Horizontally stack groups of 2 and 4 items with the same
        dimensions to form single superitems
        """
        assert htype in (
            "all",
            "two-width",
            "two-depth",
            "four",
        ), "Unsupported horizontal superitem type"

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
        two_superitems = []
        for slice in two_slices:
            if htype in ("all", "two-width"):
                two_superitems += [TwoHorizontalSuperitemWidth(slice)]
            elif htype in ("all", "two-depth"):
                two_superitems += [TwoHorizontalSuperitemDepth(slice)]
        logger.debug(f"Generated {len(two_superitems)} horizontal superitems with 2 items")

        # Generate 4-items horizontal superitems
        four_superitems = []
        for slice in four_slices:
            if htype in ("all", "four"):
                four_superitems += [FourHorizontalSuperitem(slice)]
        logger.debug(f"Generated {len(four_superitems)} horizontal superitems with 4 items")

        return two_superitems + four_superitems

    @classmethod
    def _drop_singles_in_horizontal(cls, superitems):
        """
        Remove single item superitems that appear in at least
        one horizontal superitem
        """
        to_remove = []
        for s in superitems:
            if isinstance(s, HorizontalSuperitem):
                ids = s.id
                for i, o in enumerate(superitems):
                    if isinstance(o, SingleItemSuperitem) and o.id[0] in ids:
                        to_remove += [i]

        for i in sorted(to_remove, reverse=True):
            superitems.pop(i)

        return superitems

    @classmethod
    def _gen_superitems_vertical(cls, superitems, max_vstacked, tol=0.7):
        """
        Divide superitems by width-depth ratio and vertically stack each group
        """
        assert tol >= 0.0, "Tolerance must be non-negative"
        assert max_vstacked > 1, "Maximum number of stacked items must be greater than 1"

        def _gen_superitems_vertical_subgroup(superitems):
            """
            Vertically stack groups of >= 2 items or superitems with the
            same dimensions to form a taller superitem
            """

            # Add the "width * depth" column and sort superitems
            # in ascending order by that dimension
            wd = [s.width * s.depth for s in superitems]
            superitems = [superitems[i] for i in utils.argsort(wd)]

            # Extract candidate groups made up of >= 2 items or superitems
            slices = []
            for s in range(2, max_vstacked + 1):
                for i in range(0, len(superitems) - (s - 1), s):
                    good = True
                    for j in range(1, s, 1):
                        if (
                            superitems[i + j].width * superitems[i + j].depth
                            >= superitems[i].width * superitems[i].depth
                        ) and (
                            superitems[i].width * superitems[i].depth
                            <= tol * superitems[i + j].width * superitems[i + j].depth
                        ):
                            good = False
                            break
                    if good:
                        slices += [tuple(superitems[i + j] for j in range(s))]

            subgroup_vertical = []
            # Generate vertical superitems
            for slice in slices:
                subgroup_vertical += [VerticalSuperitem(slice)]

            return subgroup_vertical

        wide_superitems = []
        deep_superitems = []
        for s in superitems:
            if s.width / s.depth >= 1:
                wide_superitems.append(s)
            else:
                deep_superitems.append(s)
        wide_vsubgroup = _gen_superitems_vertical_subgroup(wide_superitems)
        logger.debug(f"Generated {len(wide_vsubgroup)} Wide VerticalSuperitems")
        deep_vsubgroup = _gen_superitems_vertical_subgroup(deep_superitems)
        logger.debug(f"Generated {len(deep_vsubgroup)} Deep VerticalSuperitems")
        return wide_vsubgroup + deep_vsubgroup

    @classmethod
    def _filter_superitems(cls, superitems, pallet_dims):
        """
        Keep only those superitems that do not exceed the
        pallet capacity
        """
        return [
            s
            for s in superitems
            if s.width <= pallet_dims.width
            and s.depth <= pallet_dims.depth
            and s.height <= pallet_dims.height
        ]
