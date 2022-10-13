"""Debug configuration for 3D Bin Packing Problem execution."""
from __future__ import annotations

from pathlib import Path

from d3_bpp.config import (
    ContainerMeasures,
    ContainerTypes,
    ContainerTypesEnum,
    MinMaxTuple,
    OrderGeneratorConfiguration,
    PackingConfiguration,
    PalletMeasures,
    PalletTypes,
    PalletTypesEnum,
    ProductGeneratorConfiguration,
)
from d3_bpp.config.rotation import RotationTypesEnum

GEN_CONFIG = ProductGeneratorConfiguration(
    num_products=int(1e6),
    product_width_range=MinMaxTuple(300, 800),
    product_depth_range=MinMaxTuple(300, 800),
    product_height_range=MinMaxTuple(300, 800),
    product_weight_range=MinMaxTuple(1, 100),
    out_folder_path=Path("data", "products"),
)

ORDER_CONFIG = OrderGeneratorConfiguration(
    product_dataset_path=Path("data", "products", "debug_products.pkl"),
    num_products=MinMaxTuple(10, 100),
    product_width_range=MinMaxTuple(300, 800),
    product_depth_range=MinMaxTuple(300, 800),
    product_height_range=MinMaxTuple(300, 800),
    product_weight_range=MinMaxTuple(1, 100),
    out_folder_path=Path("data", "orders"),
)

PACK_CONFIG = PackingConfiguration(
    order_dataset_path=Path("data", "orders", "debug_products", "debug_order.pkl"),
    container_type=ContainerTypesEnum.twenty_foot,
    pallet_type=PalletTypesEnum.eur_1,
    rotation_type=RotationTypesEnum.no_rotation,
    out_folder_path=Path("data", "packings"),
)
