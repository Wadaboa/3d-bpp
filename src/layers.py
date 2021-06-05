import numpy as np

from . import utils


class Layer:
    """
    A layer represents the placement of a collection of
    items or superitems having similar heights
    """

    def __init__(self, height, superitem_ids, coords):
        self.height = height
        self.superitem_ids = superitem_ids
        self.coords = coords

    def get_superitems(self, superitems):
        return superitems.iloc[self.superitem_ids]

    def get_item_ids(self, superitems):
        return list(
            utils.flatten(self.get_superitems(superitems).flattened_items.to_list())
        )

    def get_items(self, order, superitems):
        item_ids = self.get_item_ids(superitems)
        return order.iloc[item_ids]

    def get_volume(self, superitems):
        return self.get_superitems(superitems).volume.sum()

    def get_area(self, superitems):
        s = self.get_superitems(superitems)
        return (s.lenght * s.width).sum()

    def get_density(self, superitems, W, D, two_dims=False):
        return (
            self.get_volume(superitems) / W * D * self.height
            if two_dims
            else self.get_area(superitems) / W * D
        )

    def map_superitem_ids(self, mapping):
        self.superitem_ids = [mapping[s] for s in self.superitem_ids]

    def __str__(self):
        return f"Layer(height={self.height}, ids={self.superitem_ids})"

    def __repr__(self):
        return self.__str__()


class LayerPool:
    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        assert isinstance(
            layer, Layer
        ), "The given layer should be an instance of the Layer class"
        self.layers.append(layer)

    def add_pool(self, layer_pool):
        assert isinstance(
            layer_pool, LayerPool
        ), "The given set of layers should be an instance of the LayerPool class"
        self.layers.extend(layer_pool.layers)

    def map_superitem_ids(self, mapping):
        for layer in self.layers:
            layer.map_superitem_ids(mapping)

    def is_present(self, layer):
        assert isinstance(
            layer, Layer
        ), "The given layer should be an instance of the Layer class"
        present = False
        for other_layer in self.layers:
            if np.array_equal(
                layer.superitem_ids, other_layer.superitem_ids
            ) and np.array_equal(layer.coords, other_layer.coords):
                present = True
                break
        return present

    def get_item_ids(self, superitems):
        item_ids = []
        for layer in self.layers:
            item_ids += layer.get_item_ids(superitems)
        return item_ids

    def get_densities(self, superitems, W, D):
        return [
            layer.get_density(superitems=superitems, W=W, D=D, two_dims=True)
            for layer in self.layers
        ]

    def select_layers(self, superitems, W, D, min_density=0.5):
        # Sort layers by densities and keep only those with a
        # density greater than or equal to the given minimum
        densities = self.get_densities(superitems, W, D)
        sorted_densities = np.array(densities).argsort()
        sorted_layers = np.array(self.layers)[sorted_densities[::-1]]
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
        assert isinstance(
            e, Layer
        ), "The given layer should be an instance of the Layer class"
        self.layers[i] = e
