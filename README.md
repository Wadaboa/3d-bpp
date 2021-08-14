# 3D Bin Packing

<p align="center">
  <img src="assets/bin-example.png" />
</p>

This repository contains different implementations of heuristics-based and mathematical programming approaches to solve the 3D bin packing problem (3D-BPP):
1. Baseline MP model [[1]](#1)
2. Maxrects generalized to 3D (2D implementation based on [rectpack](https://github.com/secnot/rectpack))
3. Column generation MP model [[1]](#1)

All solutions are based on the concept of layers, which are collections of items having similar heights (within a pre-specified tolerance) that are thus having the same base z-coordinate. Using layers we are able to relax the 3D problem to a set of 2D ones, boosting efficiency-related metrics. Moreover, the usage of superitems (i.e. compounding similar items together in a bigger item) enables us to both lower the amount of items we are effectively dealing with and also to maximize layer density, thanks to the compact stacking of items in a superitem.

The bins building procedure is always handled in a procedural way, stacking layers on top of each other until a bin is full and opening new bins as needed. Before layers are placed in a bin, they are filtered based on a set of criteria (as described in [[1]](#1)), such as item coverage and layer density. After layers are placed in a bin, some items could still be "flying" (i.e. have zero support at their base): to solve this issue, we let items "fall" to the ground as much as possible, without allowing intersections, thus ensuring compactness and correctness.

## Solutions

Below you can find a broad explanation of each implemented solution. Every procedure can be called through the same function (`main()` in module `main.py`), by changing the `procedure` parameter (it could be `bl` for baseline, `mr` for maxrects or `cg` for the column generation approach). Down below you can find a MWE on how to use the library:
```python
from src import config, dataset, main

# Load dataset
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

# Get random order
order = product_dataset.get_order(50)

# Solve bin packing using the specified procedure to get
# a pool of bins without "flying" products
bin_pool = main.main(
    order,
    procedure="bl",
)

# Plot the bin pool
bin_pool.plot()
```

### Baseline

The baseline model directly assigns superitems to layers and positions them by taking into account overlapment and layer height minimization. It reproduces model [SPPSI] of [[1]](#1). Beware that it might be very slow and we advice using it only for orders under 30 items.

<p align="center">
  <img src="assets/sppsi.png" />
</p>

### Maxrects
(add citation)

Maxrects is an heuristic algorithm that was built to solve the 2D bin packing problem. In our solution, maxrects is generalized to 3D thanks to the use of layers and height groups. In particular, items are grouped together based on their 3rd dimension (the height) and a specified tolerance. Such groups of items are then used as inputs for a custom maxrects procedure in which layers are thought of as bins. The maxrects algorithm itself is implemented in the [rectpack](https://github.com/secnot/rectpack) library.

### Column generation

## Utilities

Martello's lower bounds (add citation)

## Installation

In order to install all the dependecies required by the project, you can either rely on `pip` or `conda`. If you want to use the former, `cd` to the root folder of the project and run the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r init/requirements.txt
```

Instead, for the latter you'd have to run the following commands (after changing directory to the project's root):
```bash
conda env create --name 3d-bpp -f init/environment.yml
conda activate environment.yml
```

## Execution

The library can be used from another project, by importing the necessary sources from `src/` or it can be used as a standalone bin packing library, by leveraging the interface functions in `src/main.py`.

If you want to see a working example, please check out the `bpp.ipynb` Jupyter notebook, that contains an explanation of the dataset in use, along with a comparison of all the implemented solutions.

For a more entertaining and interactive solution, you can also run the implemented [Streamlit](https://streamlit.io/) dashboard, by simply running the following command from the root of the project:
```bash
python3 -m streamlit run src/dashboard.py
```

## References
- <a id="1">[1]</a>
  _Samir Elhedhli, Fatma Gzara and Burak Yildiz (2019)_.\
  **Three-Dimensional Bin Packing and Mixed-Case Palletization**.\
  INFORMS Journal on Optimization
- <a id="2">[2]</a>
  _Silvano Martello, David Pisinger and Daniele Vigo (1998)_.\
  **The Three-Dimensional Bin Packing Problem**.\
  Operations Research
