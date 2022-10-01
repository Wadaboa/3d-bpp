import os

import numpy as np
import pandas as pd


class ProductDataset:
    """
    Construct a dataset of rectangular products,
    given minimum and maximum values for each dimension

    Samir Elhedhli, Fatma Gzara, Burak Yildiz,
    "Three-Dimensional Bin Packing and Mixed-Case Palletization",
    INFORMS Journal on Optimization, 2019.
    """

    def __init__(
        self,
        products_path,
        num_products,
        min_width,
        max_width,
        min_depth,
        max_depth,
        min_height,
        max_height,
        min_weight,
        max_weight,
        force_overload=False,
    ):
        self.products_path = products_path
        self.num_products = num_products
        self.min_width, self.max_width = min_width, max_width
        self.min_depth, self.max_depth = min_depth, max_depth
        self.min_height, self.max_height = min_height, max_height
        self.min_weight, self.max_weight = min_weight, max_weight
        self.products = self._load_products(force_overload)

    def _load_products(self, force_overload=False):
        """
        Either load the products from the given path or
        generate the dataset (if forced or if the path does not exist)
        """
        if not os.path.exists(self.products_path) or force_overload:
            products = self._gen_products()
            products.to_pickle(self.products_path)
        else:
            products = pd.read_pickle(self.products_path)
        return products

    def _gen_products(self):
        """
        Generate a sample of products, by reproducing distributions
        reported on the cited paper
        """
        # Define ratios and volumes as specified in the paper
        dw_ratios = np.random.normal(loc=0.695, scale=0.118, size=(self.num_products, 1))
        hw_ratios = np.random.lognormal(mean=-0.654, sigma=0.453, size=(self.num_products, 1))
        volumes = np.random.lognormal(mean=2.568, sigma=0.705, size=(self.num_products, 1)) * 1e6

        # Generate each dimension separately
        widths = np.clip(
            np.power(volumes / (dw_ratios * hw_ratios), 1 / 3),
            self.min_width,
            self.max_width,
        )
        depths = np.clip(widths * dw_ratios, self.min_depth, self.max_depth)
        heights = np.clip(widths * hw_ratios, self.min_height, self.max_height)
        weights = np.clip(
            np.random.lognormal(mean=2, sigma=2, size=(self.num_products, 1)),
            self.min_weight,
            self.max_weight,
        )

        # Repeat products with the given frequency of occurrence
        dims = np.concatenate((widths, depths, heights, weights), axis=1).astype(int)
        frequencies = np.ceil(
            np.random.lognormal(mean=0.544, sigma=0.658, size=(self.num_products,))
        ).astype(int)
        dims = np.repeat(dims, frequencies, axis=0)
        indexes = np.arange(0, len(dims), 1)
        dims = dims[np.random.choice(indexes, size=self.num_products)]

        # Create a DataFrame with the generated info
        df = pd.DataFrame(dims, columns=["width", "depth", "height", "weight"])
        df["volume"] = df.width * df.depth * df.height
        return df

    def get_mean_std_volumes(self):
        """
        Randomly sample (in a uniform way) from each volume category, with
        the specified sizes, in order to obtain mean and standard deviation
        statistics of the dataset used in the paper
        """
        category_one = np.random.uniform(low=2.72, high=12.04, size=72037)
        category_two = np.random.uniform(low=12.05, high=20.23, size=55436)
        category_three = np.random.uniform(low=20.28, high=32.42, size=26254)
        category_four = np.random.uniform(low=32.44, high=54.08, size=9304)
        category_five = np.random.uniform(low=54.31, high=100.21, size=3376)
        volumes = np.concatenate(
            (category_one, category_two, category_three, category_four, category_five)
        )
        log_volumes = np.log(volumes)
        return log_volumes.mean(), log_volumes.std()

    def get_order(self, ordered_products):
        """
        Sample the given number of products from the dataset
        to generate an order
        """
        order = self.products.sample(ordered_products, replace=True)
        ids = pd.Series(order.index, name="id")
        return pd.concat([ids, order.reset_index(drop=True)], axis=1)

    def get_dummy_order(self, ordered_products, dim="height"):
        """
        Return a dummy order with products having one equal dimension
        """
        order = self.get_order(ordered_products)
        order[dim] = order.sample(1)[dim].item()
        order["volume"] = order.width * order.depth * order.height
        return order
