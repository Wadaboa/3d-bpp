import numpy as np
import pandas as pd

import utils, superitems, maxrects, layers


class Bin:
    """
    A bin is a collection of layers
    """

    def __init__(self, layer_pool, pallet_dims):
        self.layer_pool = layer_pool
        self.pallet_dims = pallet_dims

    @property
    def height(self):
        """
        Return the current height of the bin
        """
        return sum(l.height for l in self.layer_pool)

    @property
    def volume(self):
        """
        Return the volume of occupied space in the bin
        """
        return sum(l.volume for l in self.layer_pool)

    @property
    def remaining_height(self):
        """
        Return the height remaining to fill up the bin
        """
        return self.pallet_dims.height - self.height

    def add(self, layer):
        """
        Add the given layer to the current bin
        """
        assert isinstance(
            layer, layers.Layer
        ), "The given layer should be an instance of the Layer class"
        self.layer_pool.add(layer)

    def get_layer_zs(self):
        """
        Return a list containing the height base coordinate for each layer in the bin
        """
        heights = [0]
        for layer in self.layer_pool[:-1]:
            heights += [heights[-1] + layer.height]
        return heights

    def get_layer_densities(self, two_dims=False):
        """
        Return the 2D/3D density of each layer in the bin
        """
        return self.layer_pool.get_densities(two_dims=two_dims)

    def get_density(self):
        """
        Return the density of the bin
        """
        return self.volume / self.pallet_dims.volume

    def sort_by_densities(self, two_dims=False):
        """
        Sort layers in the bin by decreasing density
        """
        self.layer_pool.sort_by_densities(two_dims=two_dims)

    def plot(self):
        """
        Plot the curret bin by plotting each layer in the
        bin and by stacking them vertically
        """
        height = 0
        ax = utils.get_pallet_plot(self.pallet_dims)
        for layer in self.layer_pool:
            ax = layer.plot(ax=ax, height=height)
            height += layer.height
        return ax

    def to_dataframe(self):
        """
        Return a Pandas DataFrame representation of the bin
        """
        return self.layer_pool.to_dataframe(zs=self.get_layer_zs())

    def __str__(self):
        return f"Bin({self.layer_pool})"

    def __repr__(self):
        return self.__str__()


class BinPool:
    """
    A pool of bins is a collection of bins
    """

    def __init__(self, layer_pool, pallet_dims, singles_removed=None, two_dims=False, area_tol=1.0):
        self.layer_pool = layer_pool
        self.pallet_dims = pallet_dims

        # Build the bin pool and place uncovered items on top
        # or in a new bin
        self.layer_pool.sort_by_densities(two_dims=two_dims)
        self.bins = self._build(self.layer_pool)
        self._place_not_covered(singles_removed=singles_removed, area_tol=area_tol)

        # Sort layers in each bin by density
        for bin in self.bins:
            bin.sort_by_densities(two_dims=two_dims)

    def _build(self, layer_pool):
        """
        Iteratively build the bin pool by placing
        the given layers
        """
        bins = []
        for i, layer in enumerate(layer_pool):
            placed = False

            # Place the layer in an already opened bin
            for bin in bins:
                if bin.height + layer.height <= self.pallet_dims.height:
                    bin.add(layer)
                    placed = True

            # Open a new bin
            if not placed:
                bins += [Bin(layer_pool.subset([i]), self.pallet_dims)]

        return bins

    def _place_not_covered(self, singles_removed=None, area_tol=1.0):
        """
        Place the remaining items (not superitems) either on top
        of existing bins or in a whole new bin, if they do not fit
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
            to_place = []
            for s in superitems_list:
                last_layer_area = working_bin.layer_pool[-1].area
                max_area = np.clip(area_tol * last_layer_area, 0, self.pallet_dims.area)
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
            assert len(to_place) > 0, "The number of superitems to place must be > 0"
            spool = superitems.SuperitemPool(superitems=to_place)
            layer = maxrects.maxrects_single_layer_online(spool, self.pallet_dims)
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

        # Get single superitems that are not yet covered
        # (assuming that the superitems pool in the layer pool contains all single superitems)
        superitems_list = self.layer_pool.not_covered_single_superitems(
            singles_removed=singles_removed
        )

        # Sort superitems by ascending height
        superitems_list = [
            superitems_list[i] for i in utils.argsort([s.height for s in superitems_list])
        ]

        # Get placeable and unplaceable items
        remaining_heights = self.get_remaining_heights()
        max_remaining_height = 0 if len(remaining_heights) == 0 else max(remaining_heights)
        superitems_list, remaining_items = _get_unplaceable_items(
            superitems_list, max_remaining_height
        )
        superitems_list = _place_new_layers(superitems_list, remaining_heights)

        # Place unplaceable items in a new bin
        remaining_items += superitems_list
        if len(remaining_items) > 0:
            spool = superitems.SuperitemPool(superitems=remaining_items)
            lpool = maxrects.maxrects_multiple_layers(spool, self.pallet_dims, add_single=False)
            self.layer_pool.extend(lpool)
            self.bins += self._build(lpool)

    def get_heights(self):
        """
        Return the height of each bin in the pool
        """
        return [b.height for b in self.bins]

    def get_remaining_heights(self):
        """
        Return the remaining height of each bin in the pool, which is
        the difference between the maximum height of a bin and its current one
        """
        return [b.remaining_height for b in self.bins]

    def get_layer_densities(self, two_dims=False):
        """
        Return the 2D/3D densities for each layer in each bin
        """
        return [b.get_layer_densities(two_dims) for b in self.bins]

    def get_bin_densities(self):
        """
        Return the 2D/3D densities for each bin
        """
        return [b.get_density() for b in self.bins]

    def plot(self):
        """
        Return a list of figures representing bins inside the pool
        """
        axs = []
        for bin in self.bins:
            ax = bin.plot()
            ax.set_facecolor("xkcd:white")
            axs.append(ax)
        return axs

    def to_dataframe(self):
        """
        Return a Pandas DataFrame representing bins inside the pool
        """
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


class CompactBin:
    """
    A bin without the concept of layers, in which
    items are compacted to the ground
    """

    def __init__(self, bin_df, pallet_dims):
        self.pallet_dims = pallet_dims
        self.df = self._gravity(bin_df)

    def _gravity(self, bin_df):
        """
        Let items fall as low as possible without
        intersecting with other objects
        """
        for l in range(1, bin_df.layer.max() + 1):
            layer = bin_df[bin_df.layer == l]
            for i, item in layer.iterrows():
                items_below = bin_df[bin_df.z < item.z]
                zs = [
                    prev_item.z.item() + prev_item.height.item()
                    for _, prev_item in items_below.iterrows()
                    if utils.do_overlap(item, prev_item)
                ]
                new_z = max(zs) if len(zs) > 0 else 0
                bin_df.at[i, "z"] = new_z
        return bin_df

    def plot(self):
        """
        Return a bin plot without the layers representation
        """
        ax = utils.get_pallet_plot(self.pallet_dims)
        for _, item in self.df.iterrows():
            ax = utils.plot_product(
                ax,
                item["item"],
                utils.Coordinate(item.x, item.y, item.z),
                utils.Dimension(item.width, item.depth, item.height),
            )
        return ax


class CompactBinPool:
    """
    A collection of compact bins
    """

    def __init__(self, bin_pool):
        self.compact_bins = []
        self._original_bin_pool = bin_pool
        for bin in bin_pool:
            self.compact_bins.append(CompactBin(bin.to_dataframe(), bin_pool.pallet_dims))

    def get_original_bin_pool(self):
        """
        Return the uncompacted bin pool
        """
        return self._original_bin_pool

    def get_original_layer_pool(self):
        """
        Return the layer pool used to build bins prior to compacting
        """
        return self._original_bin_pool.layer_pool

    def plot(self):
        """
        Return a list of figures representing bins inside the pool
        """
        axs = []
        for bin in self.compact_bins:
            ax = bin.plot()
            axs.append(ax)
        return axs

    def to_dataframe(self):
        """
        Return a Pandas DataFrame representing bins inside the pool
        """
        dfs = []
        for i, compact_bin in enumerate(self.compact_bins):
            df = compact_bin.df
            df["bin"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0)
