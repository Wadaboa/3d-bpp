import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import utils


class Bin:
    """
    A bin is a collection of layers
    """

    def __init__(self, layer_pool):
        self.layer_pool = layer_pool

    def add(self, layer):
        self.layer_pool.add(layer)

    @property
    def height(self):
        return sum(l.height for l in self.layer_pool)

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

    def get_pallet_plot(self, pallet_dims):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.text(0, 0, 0, "origin", size=10, zorder=1, color="k")
        ax.view_init(azim=60)
        pallet_width, pallet_depth, pallet_height = pallet_dims
        ax.set_xlim3d(0, pallet_width)
        ax.set_ylim3d(0, pallet_depth)
        ax.set_zlim3d(0, pallet_height)
        return ax

    def plot(self, pallet_dims):
        height = 0
        ax = self.get_pallet_plot(pallet_dims)
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
        layer_pool.sort_by_densities(pallet_dims[0], pallet_dims[1])
        self.pallet_dims = pallet_dims
        self.bins = self._build()

    def _build(self):
        """
        Iteratively build the bin pool by placing
        the given layers
        """
        bins = []
        _, _, pallet_height = self.pallet_dims
        for i, layer in enumerate(self.layer_pool):
            placed = False

            # Place the layer in an already opened bin
            for bin in bins:
                if bin.height + layer.height <= pallet_height:
                    bin.add(layer)
                    placed = True

            # Open a new bin
            if not placed:
                bins += [Bin(self.layer_pool.subset([i]))]

        return bins

    def plot(self):
        for bin in self.bins:
            ax = bin.plot(self.pallet_dims)
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
        assert isinstance(
            e, Bin
        ), "The given bin should be an instance of the Bin class"
        self.bins[i] = e
