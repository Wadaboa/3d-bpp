import copy

import numpy as np
import pandas as pd

from . import utils, superitems, utils


class Layer:
    """
    A layer represents the placement of a collection of
    items or superitems having similar heights
    """

    def __init__(self, height, superitems_pool, superitems_coords):
        self.height = int(height)
        self.superitems_pool = superitems_pool
        self.superitems_coords = superitems_coords

    def get_items_coords(self, z=0):
        """
        Return a dictionary having as key the item id and as value
        the item coordinates in the layer
        """
        items_coords = dict()
        for s, c in zip(self.superitems_pool, self.superitems_coords):
            coords = s.get_items_coords(width=c.x, depth=c.y, height=z)
            if utils.duplicate_keys([items_coords, coords]):
                print("Duplicated item in the same layer")
            items_coords = {**items_coords, **coords}
        return items_coords

    def get_items_dims(self):
        items_dims = dict()
        for s, c in zip(self.superitems_pool, self.superitems_coords):
            dims = s.get_items_dims()
            if utils.duplicate_keys([items_dims, dims]):
                print("Duplicated item in the same layer")
            items_dims = {**items_dims, **dims}
        return items_dims

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer
        """
        return self.superitems_pool.get_unique_item_ids()

    def get_density(self, W, D, two_dims=False):
        """
        Compute the 2D/3D density of the layer
        """
        return self.volume / (W * D * self.height) if not two_dims else self.area / (W * D)

    def remove(self, superitem):
        """
        Remove the given superitem from the layer
        """
        self.superitems_pool.remove(superitem)
        del#

    def pop(self, i):
        """
        Remove the superitem at the given index from the layer
        """
        self.superitems_pool.pop(i)
        self.superitems_coords.pop(i)

    def get_superitems_containing_item(self, item_id):
        """
        Return a list of superitems containing the given raw item
        """
        return self.superitems_pool.get_superitems_containing_item(item_id)

    @property
    def volume(self):
        return sum(s.volume for s in self.superitems_pool)

    @property
    def area(self):
        return sum(s.width * s.depth for s in self.superitems_pool)

    def to_dataframe(self, z=0):
        items_coords = self.get_items_coords()
        items_dims = self.get_items_dims()
        keys = list(items_coords.keys())
        xs = [items_coords[k].x for k in keys]
        ys = [items_coords[k].y for k in keys]
        zs = [items_coords[k].z + z for k in keys]
        ws = [items_dims[k].width for k in keys]
        ds = [items_dims[k].depth for k in keys]
        hs = [items_dims[k].height for k in keys]
        return pd.DataFrame(
            {
                "item": keys,
                "x": xs,
                "y": ys,
                "z": zs,
                "width": ws,
                "depth": ds,
                "height": hs,
            }
        )

    def rearrange(self, W, D):
        """
        Apply maxrects over superitems in layer
        """
        items_dims = self.get_items_dims()
        ws = [items_dims[k].width for k in items_dims]
        ds = [items_dims[k].depth for k in items_dims]
        placed, layer = utils.maxrects_single_layer(self.superitems_pool, ws, ds, W, D)
        assert placed, "Couldn't rearrange superitems"
        self.superitems_coords = layer.superitems_coords

    def __str__(self):
        return f"Layer(height={self.height}, ids={self.superitems_pool.get_unique_item_ids()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_items_coords() == other.get_items_coords()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.superitems_pool)

    def __contains__(self, superitem):
        return superitem in self.superitems_pool


class LayerPool:
    """
    A layer pool is a collection of layers
    """

    def __init__(self, superitems_pool, layers=None, add_single=False):
        self.superitems_pool = superitems_pool
        self.layers = layers or []
        if add_single:
            self._add_single_layers()

    def subset(self, layer_indices=None):
        """
        Return a new layer pool with the given subset of layers
        and the same superitems pool
        """
        layers = [
            l for i, l in enumerate(self.layers) if layer_indices is None or i in layer_indices
        ]
        return LayerPool(self.superitems_pool, layers=layers, add_single=False)

    def _add_single_layers(self):
        """
        Add one layer for each superitem that only
        contains that superitem
        """
        for superitem in self.superitems_pool:
            self.add(
                Layer(
                    superitem.height,
                    superitems.SuperitemPool(superitems=[superitem]),
                    [utils.Coordinate(x=0, y=0)],
                )
            )

    def get_ol(self):
        """
        Return a numpy array ol s.t. ol[l] = h iff
        layer l has height h
        """
        return np.array([layer.height for layer in self.layers], dtype=int)

    def get_zsl(self):
        """
        Return a binary matrix zsl s.t. zsl[s, l] = 1 iff
        superitem s is in layer l
        """
        zsl = np.zeros((len(self.superitems_pool), len(self.layers)), dtype=int)
        for s, superitem in enumerate(self.superitems_pool):
            for l, layer in enumerate(self.layers):
                if superitem in layer:
                    zsl[s, l] = 1
        return zsl

    def add(self, layer):
        """
        Add the given Layer to the current pool
        """
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        if layer not in self.layers:
            self.layers.append(layer)

    def extend(self, layer_pool):
        """
        Extend the current pool with the given one
        """
        assert isinstance(
            layer_pool, LayerPool
        ), "The given set of layers should be an instance of the LayerPool class"
        for layer in layer_pool:
            self.add(layer)
        self.superitems_pool.extend(layer_pool.superitems_pool)

    def pop(self, i):
        """
        Remove the layer at the given index from the pool
        """
        self.layers.pop(i)

    def add_to_superitems_pool(self, superitem):
        self.superitems_pool.add(superitem)

    def extend_superitems_pool(self, superitems_pool):
        self.superitems_pool.extend(superitems_pool)

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer pool
        """
        return self.superitems_pool.get_unique_item_ids()

    def get_densities(self, W, D, two_dims=True):
        """
        Compute the 2D/3D density of each layer in the pool
        """
        return [layer.get_density(W=W, D=D, two_dims=two_dims) for layer in self.layers]

    def sort_by_densities(self, W, D, two_dims=True):
        """
        Sort layers in the pool by decreasing density
        """
        densities = self.get_densities(W, D, two_dims=two_dims)
        sorted_indices = np.array(densities).argsort()[::-1]
        self.layers = [self.layers[i] for i in sorted_indices]

    def discard_by_densities(self, W, D, min_density=0.5, two_dims=True):
        """
        Sort layers by densities and keep only those with a
        density greater than or equal to the given minimum
        """
        self.sort_by_densities(W, D, two_dims=two_dims)
        densities = self.get_densities(W, D, two_dims=two_dims)
        last_index = -1
        for i, d in enumerate(densities):
            if d >= min_density:
                last_index = i
            else:
                break
        return self.subset(list(range(last_index + 1)))

    def discard_by_coverage(self, max_coverage=3):
        """
        Post-process layers by their item coverage
        """
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [0] * len(all_item_ids)))
        layers_to_select = []
        for i, layer in enumerate(self.layers):
            to_select = True
            already_covered = 0

            # Stop when all items are covered
            if all([c > 0 for c in item_coverage.values()]):
                break

            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                # If at least one item in the layer was already selected
                # more times than the maximum allowed value, then such layer
                # is to be discarded
                if item_coverage[item] >= max_coverage:
                    to_select = False
                    break

                # If at least `max_coverage` items in the layer are already covered
                # by previously selected layers, then such layer is to be discarded
                if item_coverage[item] > 0:
                    already_covered += 1
                if already_covered >= max_coverage:
                    to_select = False
                    break

            # If the layer is selected, increase item coverage
            # for each item in such layer and add it to the pool
            # of selected layers
            if to_select:
                layers_to_select += [i]
                for item in item_ids:
                    item_coverage[item] += 1

        return self.subset(layers_to_select)

    def remove_duplicated_items(self, W, D, min_density=0.5, two_dims=True):
        """
        Keep items that are covered multiple times only
        in the layers with the highest densities
        """
        selected_layers = copy.deepcopy(self)
        all_item_ids = selected_layers.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        edited_layers = set()
        to_remove = []
        for l, layer in enumerate(selected_layers.layers):
            # Remove duplicated superitems
            for item in item_coverage.keys():
                if item_coverage[item]:
                    for s in layer.get_superitems_containing_item(item):
                        layer.remove(s)
                        edited_layers.add(l)

            # Flag the layer if it doesn't respect the minimum density,
            # or update item coverage otherwise
            if layer.get_density(W=W, D=D, two_dims=two_dims) < min_density:
                to_remove += [l]
            else:
                item_ids = layer.get_unique_items_ids()
                for item in item_ids:
                    item_coverage[item] = True

        # Rearrange layers in which at least one superitem was removed
        for l in edited_layers:
            if l not in to_remove:
                selected_layers[l].rearrange(W, D)

        # Remove edited layers that do not respect the minimum
        # density requirement after removing at least one superitem
        for l in sorted(to_remove, reverse=True):
            selected_layers.pop(l)

        return selected_layers

    def remove_empty_layers(self):
        """
        Check and remove layers without any items
        """
        selected_layers = copy.deepcopy(self)
        not_empty_layers = []
        for l, layer in enumerate(selected_layers.layers):
            if layer.volume != 0:
                not_empty_layers.append(l)
        return selected_layers.subset(not_empty_layers)

    def select_layers(
        self, W, D, min_density=0.5, two_dims=True, max_coverage=3, remove_duplicated=True
    ):
        """
        Perform post-processing steps to select the best layers in the pool
        """
        new_pool = self.discard_by_densities(W, D, min_density=min_density, two_dims=two_dims)
        new_pool = new_pool.discard_by_coverage(max_coverage=max_coverage)
        if remove_duplicated:
            new_pool = new_pool.remove_duplicated_items(W, D)
        new_pool = new_pool.remove_empty_layers()
        new_pool.sort_by_densities(W, D, two_dims=two_dims)
        return new_pool

    def item_coverage(self):
        """
        Return a dictionary {i: T/F} identifying whether or not
        item i is included in a layer in the pool
        """
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        for layer in self.layers:
            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                item_coverage[item] = True
        return item_coverage

    def not_covered_single_superitems(self):
        item_coverage = self.item_coverage()
        not_covered_ids = [k for k, v in item_coverage.items() if not v]
        not_covered = set()
        for s in self.superitems_pool:
            for i in not_covered_ids:
                if s.id == [i]:
                    not_covered.add(s)
        return list(not_covered)

    def to_dataframe(self, zs=None):
        if zs is None:
            zs = [0] * len(self)
        dfs = []
        for i, layer in enumerate(self.layers):
            df = layer.to_dataframe(z=zs[i])
            df["layer"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0).reset_index(drop=True)

    def __str__(self):
        return f"LayerPool(layers={self.layers})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.layers)

    def __contains__(self, layer):
        return layer in self.layers

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, e):
        assert isinstance(e, Layer), "The given layer should be an instance of the Layer class"
        self.layers[i] = e
