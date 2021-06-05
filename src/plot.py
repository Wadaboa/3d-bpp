import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def from_blb_to_vertices(x, y, z, l, w, h):
    blb = [x, y, z]
    blf = [x + l, y, z]
    brb = [x, y + w, z]
    brf = [x + l, y + w, z]

    tlb = [x, y, z + h]
    tlf = [x + l, y, z + h]
    trb = [x, y + w, z + h]
    trf = [x + l, y + w, z + h]

    return np.array([blb, blf, brb, brf, tlb, tlf, trb, trf])


def from_vertices_to_faces(v):
    return np.array(
        [
            [v[0], v[1], v[3], v[2]],  # bottom
            [v[4], v[5], v[7], v[6]],  # top
            [v[0], v[2], v[6], v[4]],  # back
            [v[1], v[3], v[7], v[5]],  # front
            [v[0], v[1], v[5], v[4]],  # left
            [v[2], v[3], v[7], v[6]],  # right
        ]
    )


def add_product_to_pallet(ax, pid, blb, dims):
    v = from_blb_to_vertices(*blb, *dims)
    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
    f = from_vertices_to_faces(v)
    ax.add_collection3d(
        Poly3DCollection(
            f, facecolors=np.random.rand(1, 3), linewidths=1, edgecolors="r", alpha=0.45
        )
    )
    ax.text(
        blb[0] + dims[0] // 2,
        blb[1] + dims[1] // 2,
        blb[2] + dims[2] // 2,
        pid,
        size=10,
        zorder=1,
        color="k",
    )
    return ax


def get_pallet(pallet_dims):
    pallet_width, pallet_depth, pallet_height = pallet_dims
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.text(0, 0, 0, "origin", size=10, zorder=1, color="k")
    ax.view_init(azim=60)
    ax.set_xlim3d(0, pallet_width)
    ax.set_ylim3d(0, pallet_depth)
    ax.set_zlim3d(0, pallet_height)
    return ax


def plot_pallet(order, ids, blbs, pallet_lenght, pallet_width, max_product_height):
    ax = get_pallet(pallet_lenght, pallet_width, max_product_height)
    for i, blb in zip(ids, blbs):
        ax = add_product_to_pallet(
            ax, i, blb, (order.iloc[i].lenght, order.iloc[i].width, order.iloc[i].height)
        )
    plt.show()
    return ax
