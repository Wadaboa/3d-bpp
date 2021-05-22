import os

import numpy as np
import pandas as pd


def get_mean_std_volumes():
	category_one = np.random.uniform(low=2.72, high=12.04, size=72037)
	category_two = np.random.uniform(low=12.05, high=20.23, size=55436)
	category_three = np.random.uniform(low=20.28, high=32.42, size=26254)
	category_four = np.random.uniform(low=32.44, high=54.08, size=9304)
	category_five = np.random.uniform(low=54.31, high=100.21, size=3376)
	volumes = np.concatenate((category_one, category_two, category_three, category_four, category_five))
	log_volumes = np.log(volumes)
	return log_volumes.mean(), log_volumes.std()


class ProductDataset():

	def __init__(self, 
				 products_path, 
				 num_products, 
				 min_lenght, max_lenght, 
				 min_width, max_width, 
				 min_height, max_height, 
				 min_weight, max_weight, 
				 force_overload=False):
		self.products_path = products_path
		self.num_products = num_products  
		self.min_lenght, self.max_lenght = min_lenght, max_lenght
		self.min_width, self.max_width = min_width, max_width
		self.min_height, self.max_height = min_height, max_height
		self.min_weight, self.max_weight = min_weight, max_weight
		self.products = self._load_products(force_overload)
	
	def _load_products(self, force_overload=False):
		if not os.path.exists(self.products_path) or force_overload:
			products = self._gen_products()
			products.to_pickle(self.products_path)
		else:
			products = pd.read_pickle(self.products_path)
		return products

	def _gen_products(self):
		lw_ratios = np.random.normal(loc=0.695, scale=0.118, size=(self.num_products, 1))
		hw_ratios = np.random.lognormal(mean=-0.654, sigma=0.453, size=(self.num_products, 1))
		volumes = np.random.lognormal(mean=2.568, sigma=0.705, size=(self.num_products, 1)) * 1e+6

		widths = np.clip(np.power(volumes / (lw_ratios * hw_ratios), 1 / 3), self.min_width, self.max_width)
		heights = np.clip(widths * hw_ratios, self.min_height, self.max_height)
		lenghts = np.clip(widths * lw_ratios, self.min_lenght, self.max_lenght)
		weights = np.clip(np.random.lognormal(mean=2, sigma=2, size=(self.num_products, 1)), self.min_weight, self.max_weight)
		
		dims = np.concatenate((lenghts, widths, heights, weights), axis=1).astype(int)
		frequencies = np.ceil(np.random.lognormal(mean=0.544, sigma=0.658, size=(self.num_products, ))).astype(int)
		dims = np.repeat(dims, frequencies, axis=0)

		indexes = np.arange(0, len(dims), 1)
		dims = dims[np.random.choice(indexes, size=self.num_products)]

		df = pd.DataFrame(dims, columns=["lenght", "width", "height", "weight"])
		df["volume"] = df.lenght * df.width * df.height
		return df

	def get_order(self, ordered_products):
		order = self.products.sample(ordered_products, replace=True)
		ids = pd.Series(order.index, name="id")
		return pd.concat([ids, order.reset_index(drop=True)], axis=1)
