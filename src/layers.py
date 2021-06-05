import numpy as np
import pandas as pd

from . import utils, superitems


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

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer
        """
        return self.superitems_pool.get_unique_item_ids()

    def get_density(self, W, D, two_dims=False):
        """
        Compute the 2D/3D density of the layer
        """
        return self.volume / W * D * self.height if not two_dims else self.area / W * D

    @property
    def volume(self):
        return sum(s.volume for s in self.superitems_pool)

    @property
    def area(self):
        return sum(s.width * s.depth for s in self.superitems_pool)

    def to_dataframe(self):
        items_coords = self.get_items_coords()
        keys = list(items_coords.keys())
        xs = [items_coords[k].x for k in keys]
        ys = [items_coords[k].y for k in keys]
        zs = [items_coords[k].z for k in keys]
        return pd.DataFrame({"item": keys, "x": xs, "y": ys, "z": zs})

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

    def __init__(self, superitems_pool, layers=None, add_single=True):
        self.superitems_pool = superitems_pool
        self.layers = layers or []
        if add_single:
            self._add_single_layers()

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
        self.layers.append(layer)

    def extend(self, layer_pool):
        """
        Extend the current pool with the given one
        """
        assert isinstance(
            layer_pool, LayerPool
        ), "The given set of layers should be an instance of the LayerPool class"
        self.layers.extend(layer_pool)
        self.superitems_pool.extend(layer_pool.superitems_pool)

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer pool
        """
        return sorted(set(utils.flatten([layer.get_unique_items_ids() for layer in self.layers])))

    def get_densities(self, W, D, two_dims=True):
        """
        Compute the 2D/3D density of each layer in the pool
        """
        return [layer.get_density(W=W, D=D, two_dims=two_dims) for layer in self.layers]

    def select_layers(self, W, D, min_density=0.5, two_dims=True):
        # Sort layers by densities and keep only those with a
        # density greater than or equal to the given minimum
        densities = self.get_densities(W, D, two_dims=two_dims)
        sorted_densities = np.array(densities).argsort()
        sorted_layers = np.array(self.layers)[sorted_densities[::-1]]
        sorted_layers = LayerPool(
            self.superitems_pool,
            layers=[l for d, l in zip(densities, sorted_layers) if d >= min_density],
        )

        # Discard layers after all items are covered
        all_item_ids = sorted_layers.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        selected_layers = LayerPool(self.superitems_pool)
        for layer in sorted_layers:
            # Stop when all items are covered
            if all(list(item_coverage.values())):
                break

            # Update coverage
            item_ids = layer.get_unique_items_ids()
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

    def to_dataframe(self):
        dfs = []
        for i, layer in enumerate(self.layers):
            df = layer.to_dataframe()
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
