import os
import sys
from collections import namedtuple
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import altair as alt
from ortools.sat.python import cp_model
from matplotlib import pyplot as plt
from tqdm import tqdm

import config, dataset, utils, bins, main


def get_altair_hist_plot(series, name, bin_min, bin_max, bin_step):
    hist, bin_edges = np.histogram(
        series,
        bins=np.arange(bin_min, bin_max, bin_step),
    )
    print(bin_edges)
    data = pd.DataFrame({name: bin_edges[:-1], "Count": hist})
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            alt.X(f"{name}:Q", bin=alt.Bin(extent=[bin_min, bin_max], step=bin_step)), y="Count"
        )
    )


st.header("3D Bin Packing")

st.subheader("Dataset")
product_dataset = dataset.ProductDataset(
    "data/products.pkl",
    config.NUM_PRODUCTS,
    config.MIN_PRODUCT_WIDTH,
    config.MAX_PRODUCT_WIDTH,
    config.MIN_PRODUCT_DEPTH,
    config.MAX_PRODUCT_DEPTH,
    config.MIN_PRODUCT_HEIGHT,
    config.MAX_PRODUCT_HEIGHT,
    config.MIN_PRODUCT_WEIGHT,
    config.MAX_PRODUCT_WEIGHT,
    force_overload=False,
)

dw_ratio_plot = get_altair_hist_plot(
    product_dataset.products.depth / product_dataset.products.width,
    "D/W Ratio",
    0,
    10,
    0.5,
)
st.altair_chart(dw_ratio_plot, use_container_width=True)

hw_ratio_plot = get_altair_hist_plot(
    product_dataset.products.height / product_dataset.products.width,
    "H/W Ratio",
    0,
    10,
    0.5,
)
st.altair_chart(hw_ratio_plot, use_container_width=True)

volume_plot = get_altair_hist_plot(product_dataset.products.volume / 1e6, "Volume", 0, 100, 1)
st.altair_chart(volume_plot, use_container_width=True)

weight_plot = get_altair_hist_plot(product_dataset.products.weight, "Weight", 0, 100, 5)
st.altair_chart(weight_plot, use_container_width=True)


st.subheader("Order")
ordered_products = st.slider("Ordered products", 0, 1000, value=10, step=5)
order = product_dataset.get_order(ordered_products)
st.dataframe(order)


st.subheader("Solution")
solution_type = st.selectbox(
    "Select the algorithm you'd like to test",
    ("Baseline", "Maxrects", "Column generation"),
    index=1,
)

st.write("Computed layers:")
if solution_type == "Baseline":
    layer_pool = main.baseline_procedure(order)
elif solution_type == "Maxrects":
    layer_pool = main.maxrect_procedure(order)
st.dataframe(layer_pool.to_dataframe())

st.write("Final bins:")
bin_pool = bins.BinPool(layer_pool, config.PALLET_DIMS)
st.dataframe(bin_pool.to_dataframe())
axs = bin_pool.plot()
for i, bin in enumerate(bin_pool):
    st.write(f"Bin #{i + 1}:")
    st.write(f"* Height: {bin.height}")
    for j, layer in enumerate(bin.layer_pool):
        st.write(
            f"* Layer #{j}: Height: {layer.height} / Area: {layer.area} / Volume: {layer.volume} / 2D Density: {layer.get_density(two_dims=True)} / 3D Density: {layer.get_density(two_dims=False)}"
        )
    ax = bin.plot()
    st.pyplot(plt.gcf())
st.balloons()
