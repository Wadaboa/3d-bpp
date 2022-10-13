"""Module to manage bins and their construction."""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.axes as mtplt_axes
import numpy as np
import pandas as pd

from .layers import Layer, LayerPool
from .maxrects import maxrects_multiple_layers, maxrects_single_layer_online
from .superitems import Superitem, SuperitemPool
from .utils import Coordinate, Dimension, argsort, get_pallet_plot, plot_product


@dataclass
class Bin:
    """A bin is a collection of layers."""

    layers: LayerPool
    """Layer pool inside of the bin."""
    pallet_dims: Dimension
    """Pallet dimensions."""

    @property
    def height(self) -> int:
        """Return the current height of the bin."""
        return sum(layer.height for layer in self.layers)

    @property
    def volume(self) -> int:
        """Return the volume of occupied space in the bin."""
        return sum(layer.volume for layer in self.layers)

    @property
    def remaining_height(self) -> int:
        """Return the height remaining to fill up the bin."""
        return self.pallet_dims.height - self.height

    @property
    def density(self) -> float:
        """Return the volume density of the bin w.r.t the pallet volume."""
        return self.volume / self.pallet_dims.volume

    def add(self, layer: Layer) -> None:
        """Add the given layer to the current bin."""
        self.layers.add(layer)

    def layers_zs(self) -> list[int]:
        """Return a list containing the height base coordinate for each layer in the bin.

        The heights are from the bottom of the bin to the top of the bin.
        Each layer height is the sum of the heights of all the layers below it.
        """
        heights = [0]
        for layer in self.layers[:-1]:
            heights += [heights[-1] + layer.height]
        return heights

    def layers_densities(self, two_dims=False) -> list[float]:
        """Return the 2D or 3D density of each layer in the bin."""
        return self.layers.get_densities(two_dims=two_dims)

    def sort_by_densities(self, two_dims=False) -> None:
        """Sort layers in the bin by decreasing density."""
        self.layers.sort_by_densities(two_dims=two_dims)

    def plot(self) -> mtplt_axes.Axes:
        """Plot the current bin by plotting each layer and by stacking them vertically."""
        height = 0
        ax = get_pallet_plot(self.pallet_dims)
        for layer in self.layers:
            ax = layer.plot(ax=ax, height=height)
            height += layer.height
        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Return a Pandas DataFrame representation of the bin."""
        return self.layers.to_dataframe(zs=self.get_layers_zs())


@dataclass
class BinPool:
    """A pool of bins is a collection of bins."""

    layer_pool: LayerPool
    """Layer pool composed of all the layers to be placed in the bin pool, unrelated to the bins."""
    pallet_dims: Dimension
    """Pallet dimensions."""
    bins: list[Bin] = field(init=False, default_factory=list)
    """Bins in the pool."""

    def build_bins(
        self, singles_removed=None, two_dims: bool = False, area_tol: float = 1.0
    ) -> None:
        """Build the bin pool and place uncovered items on top or in a new bin."""
        self.layer_pool.sort_by_densities(two_dims=two_dims)
        self.bins = self._build(self.layer_pool)
        self._place_not_covered(singles_removed=singles_removed, area_tol=area_tol)

        # Sort layers in each bin by density
        for bin in self.bins:
            bin.sort_by_densities(two_dims=two_dims)

    def _build(self, layer_pool: LayerPool) -> list[Bin]:
        """Iteratively build the bin pool by placing the given layers."""
        bins: list[Bin] = []
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

    def _place_not_covered(self, singles_removed=None, area_tol: float = 1.0) -> None:
        """Place the remaining items (not superitems).

        The items can be placed either on top of existing bins or in a whole new bin,
        if they do not fit in the remaining space of the existing bins.
        """

        def _get_unplaceable_items(
            super_items_list: list[Superitem], max_spare_height: int
        ) -> tuple[list[Superitem], list[Superitem]]:
            """Separate super items into two subgroups.

            Items which height fit into existing bins.
            Items which height doesn't fit and must be placed in a new bin.

            Args:
                super_items_list: List of super items to be placed.
                    The super items must be sorted by decreasing height.
                max_spare_height: Maximum height of the remaining space in the existing bins.

            Returns:
                A tuple containing two lists of super items.
                The first list contains the super items which height fit into existing bins.
                The second list contains the super items which height doesn't fit and must be
                placed in a new bin.
            """
            index = len(super_items_list)
            for i, s in enumerate(super_items_list):
                if s.height > max_spare_height:
                    index = i
                    break
            return super_items_list[:index], super_items_list[index:]

        def _get_placeable_items(superitems_list, working_bin):
            """Return items that can be placed in a new layer in the given bin."""
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

        def _get_new_layer(to_place: list[Superitem]) -> Layer:
            """Place the maximum amount of items that can fit in a new layer."""
            if len(to_place) == 0:
                raise ValueError("No superitems to place in a new layer.")
            spool = SuperitemPool(superitems=to_place)
            layer = maxrects_single_layer_online(spool, self.pallet_dims)
            return layer

        def _place_new_layers(superitems_list, remaining_heights: list[int]):
            """Try to place items in the bin with the least spare height.

            If the items do not fit in the bin, fallback to the other open bins.
            """
            sorted_indices = argsort(remaining_heights)
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
        superitems_list = [superitems_list[i] for i in argsort([s.height for s in superitems_list])]

        # Get placeable and unplaceable items
        max_remaining_height = max(self.remaining_heights, default=0)
        superitems_list, remaining_items = _get_unplaceable_items(
            superitems_list, max_remaining_height
        )
        superitems_list = _place_new_layers(superitems_list, self.remaining_heights)

        # Place unplaceable items in a new bin
        remaining_items += superitems_list
        if len(remaining_items) > 0:
            spool = SuperitemPool(superitems=remaining_items)
            lpool = maxrects_multiple_layers(spool, self.pallet_dims, add_single=False)
            self.layer_pool.extend(lpool)
            self.bins += self._build(lpool)

    @property
    def heights(self) -> list[int]:
        """Return the height of each bin in the pool."""
        return [bin_.height for bin_ in self.bins]

    @property
    def remaining_heights(self) -> list[int]:
        """Return the remaining height of each bin in the pool.

        The remaining_height of a bin is the difference between its
        maximum pallet height w.r.t its current one.
        """
        return [bin_.remaining_height for bin_ in self.bins]

    def layers_densities(self, two_dims: bool = False) -> list[list[float]]:
        """Return the 2D or 3D densities for each layer in each bin."""
        return [bin_.layers_densities(two_dims) for bin_ in self.bins]

    def bin_densities(self) -> list[float]:
        """Return the 2D or 3D densities for each bin."""
        return [b.density for b in self.bins]

    def plot(self) -> list[mtplt_axes.Axes]:
        """Return a list of figures representing bins inside the pool."""
        axs = []
        for bin_ in self.bins:
            ax = bin_.plot()
            ax.set_facecolor("xkcd:white")
            axs.append(ax)
        return axs

    def to_dataframe(self) -> pd.DataFrame:
        """Return a Pandas DataFrame representing bins inside the pool."""
        dfs = []
        for i, bin_ in enumerate(self.bins):
            df = bin_.to_dataframe()
            df["bin"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0)

    def __len__(self) -> int:
        """Return the number of bins in the pool."""
        return len(self.bins)

    def __contains__(self, bin_: Bin) -> bool:
        """Return True if the given bin is in the pool."""
        return bin_ in self.bins

    def __getitem__(self, i: int) -> Bin:
        """Return the i-th bin in the pool."""
        return self.bins[i]


@dataclass
class CompactBin:
    """A bin without the concept of layers, in which items are compacted to the ground."""

    pallet_dims: Dimension
    """The dimensions of the pallet."""
    bin_: Bin
    """The bin to compact."""

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
                    if do_overlap(item, prev_item)
                ]
                new_z = max(zs) if len(zs) > 0 else 0
                bin_df.at[i, "z"] = new_z
        return bin_df

    def plot(self):
        """
        Return a bin plot without the layers representation
        """
        ax = get_pallet_plot(self.pallet_dims)
        for _, item in self.df.iterrows():
            ax = plot_product(
                ax,
                item["item"],
                Coordinate(item.x, item.y, item.z),
                Dimension(item.width, item.depth, item.height),
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
