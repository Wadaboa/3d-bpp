from operator import add
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import utils, superitems, layers, warm_start


class Bin:
    """
    A bin is a collection of layers
    """

    def __init__(self, layer_pool, pallet_dims):
        self.layer_pool = layer_pool
        self.pallet_dims = pallet_dims

    def add(self, layer):
        self.layer_pool.add(layer)

    @property
    def height(self):
        return sum(l.height for l in self.layer_pool)

    @property
    def remaining_height(self):
        """
        Return the height remaining to fill up the bin
        """
        _, _, pallet_height = self.pallet_dims
        return pallet_height - self.height

    def get_layer_zs(self):
        heights = [0]
        for layer in self.layer_pool[:-1]:
            heights += [heights[-1] + layer.height]
        return heights

    def add_product_to_pallet(self, ax, item_id, coords, dims):
        vertices = utils.Vertices(coords, dims)
        ax.scatter3D(vertices.get_xs(), vertices.get_ys(), vertices.get_zs())
        ax.add_collection3d(
            Poly3DCollection(
                vertices.to_faces(),
                facecolors=np.random.rand(1, 3),
                linewidths=1,
                edgecolors="r",
                alpha=0.45,
            )
        )
        center = vertices.get_center()
        ax.text(
            center.x,
            center.y,
            center.z,
            item_id,
            size=10,
            zorder=1,
            color="k",
        )
        return ax

    def get_pallet_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.text(0, 0, 0, "origin", size=10, zorder=1, color="k")
        ax.view_init(azim=60)
        pallet_width, pallet_depth, pallet_height = self.pallet_dims
        ax.set_xlim3d(0, pallet_width)
        ax.set_ylim3d(0, pallet_depth)
        ax.set_zlim3d(0, pallet_height)
        return ax

    def plot(self):
        height = 0
        ax = self.get_pallet_plot()
        for layer in self.layer_pool:
            items_coords = layer.get_items_coords(z=height)
            items_dims = layer.get_items_dims()
            for item_id in items_coords.keys():
                coords = items_coords[item_id]
                dims = items_dims[item_id]
                ax = self.add_product_to_pallet(ax, item_id, coords, dims)
            height += layer.height
        return ax

    def to_dataframe(self):
        return self.layer_pool.to_dataframe(zs=self.get_layer_zs())

    def __str__(self):
        return f"Bin({self.layer_pool})"

    def __repr__(self):
        return self.__str__()


class BinPool:
    """
    A pool of bins is a collection of bins
    """

    def __init__(self, layer_pool, pallet_dims):
        self.layer_pool = layer_pool
        self.layer_pool.sort_by_densities(pallet_dims[0], pallet_dims[1])
        self.pallet_dims = pallet_dims
        self.bins = self._build(self.layer_pool)
        self._place_not_covered()

    def _build(self, layer_pool):
        """
        Iteratively build the bin pool by placing
        the given layers
        """
        bins = []
        _, _, pallet_height = self.pallet_dims
        for i, layer in enumerate(layer_pool):
            placed = False

            # Place the layer in an already opened bin
            for bin in bins:
                if bin.height + layer.height <= pallet_height:
                    bin.add(layer)
                    placed = True

            # Open a new bin
            if not placed:
                bins += [Bin(layer_pool.subset([i]), self.pallet_dims)]

        return bins

    def _place_not_covered(self, area_tol=1.0):
        """
        Place the remaining items either on top of existing bins
        or in a whole new bin, if they do not fit
        """

        def _get_unplaceable_items(superitems_list, max_spare_height):
            """
            Return items that must be placed in a new bin
            """
            index = len(superitems_list)
            for i, s in enumerate(superitems_list):
                if s.height > max_spare_height:
                    index = i
                    break
            return superitems_list[:index], superitems_list[index:]

        def _get_placeable_items(superitems_list, working_bin):
            """
            Return items that can be placed in a new layer
            in the given bin
            """
            pallet_width, pallet_depth, _ = self.pallet_dims
            to_place = []
            for s in superitems_list:
                last_layer_area = working_bin.layer_pool[-1].area
                max_area = np.clip(area_tol * last_layer_area, 0, pallet_width * pallet_depth)
                area = sum(s.area for s in to_place)
                if area < max_area and s.height < working_bin.remaining_height:
                    to_place += [s]
                else:
                    break
            return to_place

        def _get_new_layer(to_place):
            """
            Place the maximum amount of items that can fit in
            a new layer, starting from the given pool
            """
            pallet_width, pallet_depth, _ = self.pallet_dims
            spool = superitems.SuperitemPool(superitems=to_place)
            placed = False
            while not placed:
                ws, ds, _ = spool.get_superitems_dims()
                placed, layer = utils.maxrects_single_layer(
                    spool, ws, ds, pallet_width, pallet_depth
                )
                if not placed:
                    min_superitem, _ = spool.get_extreme_superitem(minimum=True, two_dims=False)
                    spool.remove(min_superitem)
            return layer

        def _place_new_layers(superitems_list, remaining_heights):
            """
            Try to place items in the bin with the least spare height
            and fallback to the other open bins, if the layer doesn't fit
            """
            sorted_indices = utils.argsort(remaining_heights)
            working_index = 0
            while len(superitems_list) > 0 and working_index < len(self.bins):
                working_bin = self.bins[sorted_indices[working_index]]
                to_place = _get_placeable_items(superitems_list, working_bin)
                if len(to_place) > 0:
                    layer = _get_new_layer(to_place)
                    self.layer_pool.add(layer)
                    working_bin.add(layer)
                    for s in layer.superitems_pool:
                        superitems_list.remove(s)
                else:
                    working_index = working_index + 1
            return superitems_list

        # Sort superitems by ascending height
        superitems_list = self.layer_pool.not_covered_single_superitems()
        superitems_list = [
            superitems_list[i] for i in utils.argsort([s.height for s in superitems_list])
        ]

        # Get placeable and unplaceable items
        remaining_heights = self.get_remaining_heights()
        superitems_list, remaining_items = _get_unplaceable_items(
            superitems_list, max(remaining_heights)
        )
        superitems_list = _place_new_layers(superitems_list, remaining_heights)

        # Place unplaceable items in a new bin
        remaining_items += superitems_list
        if len(remaining_items) > 0:
            spool = superitems.SuperitemPool(superitems=remaining_items)
            lpool = warm_start.maxrects(spool, self.pallet_dims, add_single=False)
            self.layer_pool.extend(lpool)
            self.bins += self._build(lpool)

    def get_remaining_heights(self):
        return [b.remaining_height for b in self.bins]

    def plot(self):
        for bin in self.bins:
            ax = bin.plot()
        plt.show()

    def to_dataframe(self):
        dfs = []
        for i, bin in enumerate(self.bins):
            df = bin.to_dataframe()
            df["bin"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0)

    def __str__(self):
        return f"BinPool(bins={self.bins})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.bins)

    def __contains__(self, bin):
        return bin in self.bins

    def __getitem__(self, i):
        return self.bins[i]

    def __setitem__(self, i, e):
        assert isinstance(e, Bin), "The given bin should be an instance of the Bin class"
        self.bins[i] = e
