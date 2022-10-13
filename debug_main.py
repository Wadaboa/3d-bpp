"""Debug Main for CLI execution."""
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, NewType

from loguru import logger

from d3_bpp import __name__ as d3_bpp_name
from d3_bpp import __version__ as d3_bpp_version
from d3_bpp.config.common import MinMaxTuple
from d3_bpp.config.configuration import (
    OrderGeneratorConfiguration,
    PackingConfiguration,
    ProductGeneratorConfiguration,
)
from d3_bpp.config.container import ContainerTypesEnum
from d3_bpp.config.pallet import PalletTypesEnum
from d3_bpp.config.rotation import RotationTypesEnum

ExecutionMode = NewType("ExecutionMode", str)
"""Execution modes for the main CLI."""


class MainModes:
    """Main modes for the command line interface."""

    PROD_GEN: ExecutionMode = ExecutionMode("prod_gen")
    """Production generator mode."""
    ORDER_GEN: ExecutionMode = ExecutionMode("order_gen")
    """Order generator mode."""
    PACKER: ExecutionMode = ExecutionMode("packer")
    """Packer mode."""


args_to_min_max_tuple = {
    MainModes.PROD_GEN: ("width_range", "depth_range", "height_range", "mass_range"),
    MainModes.ORDER_GEN: (
        "num_products",
        "width_range",
        "depth_range",
        "height_range",
        "mass_range",
    ),
    MainModes.PACKER: (),
}

config_rename_dict = {
    MainModes.PROD_GEN: {
        "output": "out_folder_path",
        "id": "out_id",
        "seed": "random_seed",
        "num_products": "num_products",
        "width_range": "product_width_range",
        "depth_range": "product_depth_range",
        "height_range": "product_height_range",
        "mass_range": "product_weight_range",
    },
    MainModes.ORDER_GEN: {
        "input": "product_dataset_path",
        "output": "out_folder_path",
        "id": "out_id",
        "seed": "random_seed",
        "num_products": "num_products",
        "width_range": "product_width_range",
        "depth_range": "product_depth_range",
        "height_range": "product_height_range",
        "mass_range": "product_weight_range",
        "append_products_folder": "append_products_subfolder",
    },
    MainModes.PACKER: {
        "input": "order_dataset_path",
        "output": "out_folder_path",
        "id": "out_id",
        "seed": "random_seed",
        "container_type": "container_type",
        "pallet_type": "pallet_type",
        "rotation_type": "rotation_type",
        "append_order_folder": "append_order_subfolder",
    },
}


def create_parser() -> ArgumentParser:
    """Creates the argument parser for the command line interface."""
    parser = ArgumentParser(
        description="Debug main for the 3D-BPP solver.",
        epilog="This CLI should only be used on development stages.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{d3_bpp_name} {d3_bpp_version}",
        help="Show the version and exit.",
    )
    sub_parser = parser.add_subparsers(
        title="Actions",
        description="Select the action to run",
        required=True,
        dest="action",
    )

    help_str = "Launches the 3D-BPP solver."
    packer_parser = sub_parser.add_parser(
        name=MainModes.PACKER,
        help=help_str,
        description=help_str,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _make_3dbpp_packer_parser(packer_parser)

    help_str = "Launches the Products generator."
    product_gen_parser = sub_parser.add_parser(
        name=MainModes.PROD_GEN,
        help=help_str,
        description=help_str,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _make_product_gen_parser(product_gen_parser)

    help_str = "Launches the Order generator."
    order_gen_parser = sub_parser.add_parser(
        name=MainModes.ORDER_GEN,
        help=help_str,
        description=help_str,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _make_order_gen_parser(order_gen_parser)

    return parser


def _make_product_gen_parser(parser: ArgumentParser) -> None:
    """Creates the argument parser for the Products generator."""
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="data/products",
        help="Path to the folder where to store the generated products dataset.",
    )
    parser.add_argument(
        "-id",
        "--id",
        type=str,
        help="Custom ID for the generated dataset.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "-n",
        "--num-products",
        type=int,
        default=int(1e6),
        help="Number of products to generate.",
    )
    parser.add_argument(
        "-wr",
        "--width-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the width of the products (mm).",
    )
    parser.add_argument(
        "-dr",
        "--depth-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the depth of the products (mm).",
    )
    parser.add_argument(
        "-hr",
        "--height-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the height of the products (mm).",
    )
    parser.add_argument(
        "-mr",
        "--mass-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the mass (weight) of the products (kg).",
    )


def _make_order_gen_parser(parser: ArgumentParser) -> None:
    """Creates the argument parser for the Order generator."""
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the product dataset pickle file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="data/orders",
        help="Path to the folder where to store the generated order dataset.",
    )
    parser.add_argument(
        "-a",
        "--append-products-folder",
        action="store_true",
        default=True,
        help="Whether to append the products folder to the output path.",
    )
    parser.add_argument(
        "-id",
        "--id",
        type=str,
        help="Custom ID for the generated dataset.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "-p",
        "--num-products",
        nargs=2,
        type=int,
        default=[10, 100],
        help="Range of number of products to insert in the order.",
    )
    parser.add_argument(
        "-wr",
        "--width-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the width of the products in the order (mm).",
    )
    parser.add_argument(
        "-dr",
        "--depth-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the depth of the products in the order (mm).",
    )
    parser.add_argument(
        "-hr",
        "--height-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the height of the products in the order (mm).",
    )
    parser.add_argument(
        "-mr",
        "--mass-range",
        nargs=2,
        type=int,
        default=[200, 800],
        help="Range of the mass (weight) of the products in the order (kg).",
    )


def _make_3dbpp_packer_parser(parser: ArgumentParser) -> None:
    """Creates the argument parser for the 3D-BPP solver."""
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the order dataset in pickle format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="data/orders",
        help="Path to the folder where to store the optimized packing.",
    )
    parser.add_argument(
        "-a",
        "--append-order-folder",
        action="store_true",
        default=True,
        help="Whether to append the order folder to the output path.",
    )
    parser.add_argument(
        "-id",
        "--id",
        type=str,
        help="Custom ID for the generated packing.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "-ct",
        "--container-type",
        type=str,
        choices=[ct.value for ct in ContainerTypesEnum],
        default=ContainerTypesEnum.twenty_foot.value,
    )
    parser.add_argument(
        "-pt",
        "--pallet-type",
        type=str,
        choices=[pt.value for pt in PalletTypesEnum],
        default=PalletTypesEnum.eur_1.value,
    )
    parser.add_argument(
        "-rt",
        "--rotation-type",
        type=str,
        choices=[rt.value for rt in RotationTypesEnum],
        default=RotationTypesEnum.no_rotation.value,
    )


def _build_config_dict(args: Namespace) -> dict[str, Any]:
    """Builds the configuration dictionary from the parser arguments."""
    config_dict = {}
    for key, value in vars(args).items():
        if key != "action":
            if key in args_to_min_max_tuple[args.action]:
                value = MinMaxTuple.from_iterable(value)
            config_dict[config_rename_dict[args.action][key]] = value
    return config_dict


def product_gen(config_dict: dict[str, Any]) -> None:
    """Launches the Product dataset Generator."""
    pg_config = ProductGeneratorConfiguration(**config_dict)
    print(pg_config)


def order_gen(config_dict: dict[str, Any]) -> None:
    """Launches the Order dataset Generator."""
    og_config = OrderGeneratorConfiguration(**config_dict)
    print(og_config)


def packer(config_dict: dict[str, Any]) -> None:
    """Launches the 3D-BPP solver."""
    pk_config = PackingConfiguration(**config_dict)
    print(pk_config)


def main(args: Namespace) -> None:
    """Main of the command line interface. Runs the given commands."""
    config_dict = _build_config_dict(args)
    print(config_dict)
    match args.action:
        case MainModes.PACKER:
            packer(config_dict=config_dict)
        case MainModes.PROD_GEN:
            product_gen(config_dict=config_dict)
        case MainModes.ORDER_GEN:
            order_gen(config_dict=config_dict)
        case _:
            raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during execution: {}", e)
