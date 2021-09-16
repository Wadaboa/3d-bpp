import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt

import config, dataset, main, utils


# Matplotlib params
plt.style.use("seaborn")
plt.rcParams["figure.dpi"] = 300


def get_altair_hist_plot(series, name, bin_min, bin_max, bin_step):
    """
    Plot the given Pandas Series as an histogram using Altair
    """
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


# Title section
st.set_page_config(
    page_title="3D Bin Packing",
)
st.header("3D Bin Packing")


# Dataset section
st.header("Dataset")
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

# Plot depth over width ratio in the dataset
dw_ratio_plot = get_altair_hist_plot(
    product_dataset.products.depth / product_dataset.products.width,
    "D/W Ratio",
    0,
    1,
    0.01,
)
st.altair_chart(dw_ratio_plot, use_container_width=True)

# Plot height over width ratio in the dataset
hw_ratio_plot = get_altair_hist_plot(
    product_dataset.products.height / product_dataset.products.width,
    "H/W Ratio",
    0,
    2,
    0.05,
)
st.altair_chart(hw_ratio_plot, use_container_width=True)

# Plot volume distribution in the dataset
volume_plot = get_altair_hist_plot(product_dataset.products.volume / 1e6, "Volume", 0, 100, 1)
st.altair_chart(volume_plot, use_container_width=True)

# Plot weight distribution in the dataset
weight_plot = get_altair_hist_plot(product_dataset.products.weight, "Weight", 0, 100, 5)
st.altair_chart(weight_plot, use_container_width=True)


# Order section
st.header("Order")

# Select number of products and get random order
ordered_products = st.slider("Ordered products", 0, 1000, value=10, step=5)
order = product_dataset.get_order(ordered_products)

# Show the order as a table
st.dataframe(order)

# Show lower bounds on bins for the selected order
# on the sidebar of the dashboard
lower_bound = st.sidebar.selectbox(
    f"Lower bounds for the selected {ordered_products}-products order", ("L0", "L1", "L2")
)
if lower_bound == "L0":
    lb = utils.get_l0_lb(order, config.PALLET_DIMS)
elif lower_bound == "L1":
    lb, _, _, _ = utils.get_l1_lb(order, config.PALLET_DIMS)
elif lower_bound == "L2":
    lb, _, _, _ = utils.get_l2_lb(order, config.PALLET_DIMS)
st.sidebar.write(f"Martello's {lower_bound} lower bound: {lb}")


# Solutions section
st.header("Solution")

# Select parameters
st.subheader("Parameters")
solution_type = st.selectbox(
    "Select the algorithm you'd like to test",
    ("Baseline", "Maxrects", "Column generation"),
    index=1,
)
tlim = st.slider("Time limits", 0, 100, value=10, step=5)
max_iters = st.slider("Maximum re-iterations", 0, 5, value=1, step=1)
superitems_horizontal = st.radio("Add horizontal superitems", ("Yes", "No"))

# Compute solution
if solution_type == "Baseline" or solution_type == "Maxrects":
    bin_pool = main.main(
        order,
        procedure="bl" if solution_type == "Baseline" else "mr",
        max_iters=max_iters,
        tlim=tlim,
        superitems_horizontal=True if superitems_horizontal == "Yes" else False,
    )
elif solution_type == "Column generation":
    cg_use_height_groups = st.radio(
        "Call column generation by height groups", ("Yes", "No"), index=1
    )
    cg_mr_warm_start = st.radio(
        "Use maxrects as a warm-start for column generation", ("Yes", "No"), index=1
    )
    cg_max_iters = st.slider("Column generation maximum iterations", 0, 100, value=20, step=5)
    cg_max_stag_iters = st.slider(
        "Column generation early stopping iterations", 0, 100, value=3, step=1
    )
    cg_sp_mr = st.radio(
        "Use maxrects for the pricing subproblem in column generation", ("Yes", "No"), index=1
    )
    cg_sp_np_type = st.selectbox(
        "Select the approach to use in the subproblem no-placement for column generation",
        ("MIP", "CP"),
        index=0,
    )
    cg_sp_p_type = st.selectbox(
        "Select the approach to use in the subproblem placement for column generation",
        ("Maxrects", "MIP", "CP"),
        index=0,
    )
    bin_pool = main.main(
        order,
        procedure="cg",
        max_iters=max_iters,
        tlim=tlim,
        superitems_horizontal=True if superitems_horizontal == "Yes" else False,
        cg_use_height_groups=True if cg_use_height_groups == "Yes" else False,
        cg_mr_warm_start=True if cg_mr_warm_start == "Yes" else False,
        cg_max_iters=cg_max_iters,
        cg_max_stag_iters=cg_max_stag_iters,
        cg_sp_mr=True if cg_sp_mr == "Yes" else False,
        cg_sp_np_type=cg_sp_np_type.lower(),
        cg_sp_p_type="mr" if cg_sp_p_type == "Maxrects" else cg_sp_p_type.lower(),
    )

# Show original layer pool (before compacting)
st.subheader("Original layer pool")
st.dataframe(bin_pool.get_original_layer_pool().to_dataframe())

# Show original bin pool (before compacting)
st.subheader("Original bin pool")
original_bin_pool = bin_pool.get_original_bin_pool()
for i, bin in enumerate(original_bin_pool):
    st.write(f"Bin #{i + 1}")
    st.dataframe(bin.layer_pool.describe())
    ax = bin.plot()
    st.pyplot(plt.gcf())

# Show compact bin pool
st.subheader("Compact bin pool")
for i, bin in enumerate(bin_pool.compact_bins):
    st.write(f"Bin #{i + 1}")
    ax = bin.plot()
    st.pyplot(plt.gcf())

# Success message
st.success("Bin packing procedure successfully completed")
