import os

import numpy as np
import pandas as pd


def half_gaussian(scale, size=1, reduction_factor=1):
	return np.ceil(np.abs(
		np.random.normal(scale=scale / reduction_factor, size=size)
	)).astype(int)


class ProductDataset():

	def __init__(self, products_path, num_products, max_lenght, max_width, max_height, max_weight):
		self.products_path = products_path
		self.num_products = num_products  
		self.max_lenght = max_lenght
		self.max_width = max_width
		self.max_height = max_height
		self.max_weight = max_weight
		self.products = self._load_products()
	
	def _load_products(self):
		if not os.path.exists(self.products_path):
			products = self._gen_products()
			products.to_pickle(self.products_path)
		else:
			products = pd.read_pickle(self.products_path)
		return products

	def _gen_products(self):
		lenghts = half_gaussian(self.max_lenght, size=self.num_products, reduction_factor=4).reshape(-1, 1)
		widths = half_gaussian(self.max_width, size=self.num_products, reduction_factor=4).reshape(-1, 1)
		heights = half_gaussian(self.max_height, size=self.num_products, reduction_factor=4).reshape(-1, 1)
		weights = half_gaussian(self.max_weight, size=self.num_products, reduction_factor=50).reshape(-1, 1)
		data = np.concatenate((lenghts, widths, heights, weights), axis=1)
		return pd.DataFrame(data, columns=["lenght", "width", "height", "weight"])

	def get_order(self, ordered_products):
		order = self.products.sample(ordered_products, replace=True)
		ids = pd.Series(order.index, name="id")
		return pd.concat([ids, order.reset_index(drop=True)], axis=1)
