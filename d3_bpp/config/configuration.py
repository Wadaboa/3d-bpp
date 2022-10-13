"""Configuration module for the 3D Bin Packing problem."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from .common import MinMaxTuple, Singleton
from .container import ContainerMeasures, ContainerTypes, ContainerTypesEnum
from .pallet import PalletMeasures, PalletTypes, PalletTypesEnum
from .rotation import RotationConfig, RotationTypes, RotationTypesEnum


@dataclass(kw_only=True, frozen=True)
class Configuration(ABC, metaclass=Singleton):
    """Base class for the configuration classes."""

    config_type: ClassVar[str]
    """The configuration type."""

    execution_id: str = field(init=False)
    """Execution ID."""
    out_path: Path = field(init=False)
    """Output path."""

    out_folder_path: InitVar[Path]
    """Output folder path."""
    out_id: InitVar[str | None] = None
    """User defined Output ID, overrides internal execution ID."""

    random_seed: int = 42
    """Random seed for the generator."""

    def __post_init__(self, out_folder_path: Path, out_id: str | None) -> None:
        """Set the execution ID and the output path."""
        object.__setattr__(self, "execution_id", self._get_execution_id(out_id))

        if out_folder_path.is_file():
            raise ValueError("Output folder path must be a directory.")
        object.__setattr__(self, "out_path", self._get_out_path(out_folder_path))

        if self.out_path.exists():
            raise ValueError(f"Output path already exists: {self.out_path}")
        if self.out_path.suffix == "":
            self.out_path.mkdir(parents=True)
        else:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def check_dataset_path(self, dataset_path: Path, extension: str = ".pkl") -> None:
        """Check if the dataset path is valid.

        Args:
            dataset_path (Path): Path to the dataset.
            extension (str, optional): Dataset file extension. Defaults to ".pkl".

        Raises:
            ValueError: If the dataset path is invalid.
        """
        if (
            not dataset_path.exists()
            or dataset_path.is_dir()
            or not dataset_path.suffix == extension
        ):
            raise ValueError(f"Invalid dataset path or extension: {dataset_path}")

    @abstractmethod
    def _get_execution_id(self, out_id: str | None) -> str:
        """Computes a new execution ID of the object.

        Args:
            out_id (str|None): User defined Output ID, overrides internal execution ID.

        Returns:
            str: The execution ID.
        """

    @abstractmethod
    def _get_out_path(self, out_folder_path: Path) -> Path:
        """Get the output path."""


@dataclass(kw_only=True, frozen=True)
class ProductGeneratorConfiguration(Configuration, metaclass=Singleton):
    """Class containing the configuration for the ProductDataset generator."""

    config_type: ClassVar[str] = "product_generator"
    """The configuration type."""

    num_products: int
    """Number of products to generate."""
    product_width_range: MinMaxTuple[int]
    """Product width range (mm)."""
    product_depth_range: MinMaxTuple[int]
    """Product depth range (mm)."""
    product_height_range: MinMaxTuple[int]
    """Product height range (mm)."""
    product_weight_range: MinMaxTuple[int]
    """Product weight range (kg)."""

    def __post_init__(self, out_folder_path: Path, out_id: str | None) -> None:
        """Post-initialization method."""
        check_positives(
            self.num_products,
            self.product_width_range.min,
            self.product_depth_range.min,
            self.product_height_range.min,
            self.product_weight_range.min,
        )
        super().__post_init__(out_folder_path, out_id)

    def _get_execution_id(self, out_id: str | None) -> str:
        """Computes a new execution ID of the object.

        Args:
            out_id (str|None): User defined Output ID, overrides internal execution ID.

        Returns:
            str: The execution ID.
        """
        if out_id is not None:
            return out_id
        return "products_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_out_path(self, out_folder_path: Path) -> Path:
        """Get the output path.

        Args:
            out_folder_path (Path): Output folder path.

        Returns:
            Path: Output path.
        """
        return out_folder_path / f"{self.execution_id}.pkl"


@dataclass(frozen=True)
class OrderGeneratorConfiguration(Configuration, metaclass=Singleton):
    """Class containing the configuration for the Order generator."""

    config_type: ClassVar[str] = "order_generator"
    """The configuration type."""

    product_dataset_path: Path
    """Path to the input product dataset."""
    num_products: MinMaxTuple[int]
    """Number of products range in the order."""
    product_width_range: MinMaxTuple[int]
    """Selected products width range (mm)."""
    product_depth_range: MinMaxTuple[int]
    """Selected products depth range (mm)."""
    product_height_range: MinMaxTuple[int]
    """Selected products height range (mm)."""
    product_weight_range: MinMaxTuple[int]
    """Selected products weight range (kg)."""

    append_products_subfolder: InitVar[bool] = True
    """Append a subfolder relative to the products dataset in the output path."""

    def __post_init__(  # type: ignore[override]
        self, out_folder_path: Path, out_id: str | None, append_products_subfolder: bool
    ) -> None:
        """Set the pallet and container measures."""
        check_positives(
            self.num_products.min,
            self.product_width_range.min,
            self.product_depth_range.min,
            self.product_height_range.min,
            self.product_weight_range.min,
        )
        self.check_dataset_path(self.product_dataset_path)
        if append_products_subfolder:
            out_folder_path = out_folder_path / self.product_dataset_path.stem
        super().__post_init__(out_folder_path, out_id)

    def _get_execution_id(self, out_id: str | None) -> str:
        """Computes a new execution ID of the object.

        Args:
            out_id (str|None): User defined Output ID, overrides internal execution ID.

        Returns:
            str: The execution ID.
        """
        if out_id is not None:
            return out_id
        return "order_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_out_path(self, out_folder_path: Path) -> Path:
        """Get the output path.

        Args:
            out_folder_path (Path): Output folder path.

        Returns:
            Path: Output path.
        """
        return out_folder_path / f"{self.execution_id}.pkl"


@dataclass(frozen=True)
class PackingConfiguration(Configuration, metaclass=Singleton):
    """Class containing the configuration for the 3D bin packing solver."""

    config_type: ClassVar[str] = "3d_bin_packing_solver"
    """The configuration type."""

    order_dataset_path: Path
    """Path to the input order dataset."""
    pallet: PalletMeasures = field(init=False)
    """Pallet measures used for the packing."""
    container: ContainerMeasures = field(init=False)
    """Container measures used for the packing."""
    rotation: RotationConfig = field(init=False)
    """Supported product rotation."""
    container_type: InitVar[ContainerTypesEnum]
    """Container type."""
    pallet_type: InitVar[PalletTypesEnum]
    """Pallet type."""
    rotation_type: InitVar[RotationTypesEnum]
    """Product rotation type."""

    append_order_subfolder: InitVar[bool] = True
    """Append a subfolder relative to the order in the output path."""

    def __post_init__(  # type: ignore[override]
        self,
        out_folder_path: Path,
        out_id: str | None,
        container_type: ContainerTypesEnum,
        pallet_type: PalletTypesEnum,
        rotation_type: RotationTypesEnum,
        append_order_subfolder: bool,
    ) -> None:
        """Set the pallet and container measures."""
        self.check_dataset_path(self.order_dataset_path)
        object.__setattr__(self, "container", ContainerTypes[container_type]())
        object.__setattr__(self, "pallet", PalletTypes[pallet_type](self.container))
        object.__setattr__(self, "rotation", RotationTypes[rotation_type]())
        if append_order_subfolder:
            out_folder_path = out_folder_path / self.order_dataset_path.stem
        super().__post_init__(out_folder_path, out_id)

    def _get_execution_id(self, out_id: str | None) -> str:
        """Computes a new execution ID of the object.

        Args:
            out_id (str|None): User defined Output ID, overrides internal execution ID.

        Returns:
            str: The execution ID.
        """
        if out_id is not None:
            return out_id
        return "packing_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_out_path(self, out_folder_path: Path) -> Path:
        """Get the output path.

        Args:
            out_folder_path (Path): Output folder path.

        Returns:
            Path: Output path.
        """
        return out_folder_path / f"{self.execution_id}"


def check_positives(*value: int) -> None:
    """Check if all the given values are positive integers.

    Args:
        *value (int): Values to check.

    Raises:
        ValueError: If one of the values is not positive.
    """
    for val in value:
        if val <= 0:
            raise ValueError(f"{val} must be positive.")
