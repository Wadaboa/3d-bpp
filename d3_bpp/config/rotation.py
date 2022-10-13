"""Module containing rotation configuration types."""

from __future__ import annotations

from abc import ABC
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from d3_bpp.structure import CuboidDimension
from d3_bpp.structure.cuboid import DimensionEnum

from .common import Singleton

RotationTypes: dict[str, type[RotationConfig]] = {}
"""Dictionary containing the available rotation types."""


class RotationTypesEnum(str, Enum):
    """Rotation types enumeration."""

    no_rotation = "no_rotation"
    """No rotation."""
    x_y_rotation = "x_y_rotation"
    """Horizontal rotation of the cuboid (exchanging width and depth)."""
    free_rotation = "free_rotation"
    """Free rotation of the cuboid."""


@dataclass(kw_only=True, frozen=True)
class RotationType:
    """Rotation of a cuboid."""

    width: DimensionEnum = DimensionEnum.width
    """The new cuboid width Dimension after the rotation."""
    depth: DimensionEnum = DimensionEnum.depth
    """The new cuboid depth Dimension after the rotation."""
    height: DimensionEnum = DimensionEnum.height
    """The new cuboid height Dimension after the rotation."""

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.depth == self.height or self.width in (self.depth, self.height):
            raise ValueError(
                "Rotation impossible, duplicate dimension "
                f"{self.width} - {self.depth} - {self.height}."
            )

    def get_rotation(self, cuboid: CuboidDimension) -> CuboidDimension:
        """Return the cuboid dimension after the rotation."""
        if (
            self.width == DimensionEnum.width
            and self.depth == DimensionEnum.depth
            and self.height == DimensionEnum.height
        ):
            return cuboid
        return CuboidDimension(
            width=cuboid.get_dimension(self.width),
            depth=cuboid.get_dimension(self.depth),
            height=cuboid.get_dimension(self.width),
        )


@dataclass(frozen=True, order=True)
class RotationConfig(ABC, metaclass=Singleton):
    """Rotation configuration."""

    type_: str = field(init=False)
    """Rotation type."""

    permutations: ClassVar[list[RotationType]]
    """List of possible rotations."""

    def __init_subclass__(cls) -> None:
        """Add the class to the RotationTypes dictionary."""
        if cls.type_ in RotationTypes:
            raise ValueError(f"Rotation type {cls.type_} already exists.")
        RotationTypes[cls.type_] = cls

    def get_rotations(self, cuboid: CuboidDimension) -> Generator[CuboidDimension, None, None]:
        """Return the rotations for the cuboid."""
        yield from (permutation.get_rotation(cuboid) for permutation in self.permutations)


@dataclass(frozen=True, order=True)
class NoRotation(RotationConfig):
    """No rotation configuration."""

    type_: str = "no_rotation"
    """Rotation type."""
    permutations: ClassVar[list[RotationType]] = [RotationType()]
    """List of possible rotations."""


@dataclass(frozen=True, order=True)
class XyRotation(RotationConfig):
    """Horizontal rotation configuration."""

    type_: str = "x_y_rotation"
    """Rotation type."""
    permutations: ClassVar[list[RotationType]] = [
        RotationType(),
        RotationType(width=DimensionEnum.depth, depth=DimensionEnum.width),
    ]
    """List of possible rotations."""


@dataclass(frozen=True, order=True)
class FreeRotation(RotationConfig):
    """Free rotation configuration."""

    type_: str = "free_rotation"
    """Rotation type."""
    permutations: ClassVar[list[RotationType]] = [
        RotationType(),
        RotationType(width=DimensionEnum.depth, depth=DimensionEnum.width),
        RotationType(width=DimensionEnum.height, height=DimensionEnum.width),
        RotationType(depth=DimensionEnum.height, height=DimensionEnum.depth),
        RotationType(
            width=DimensionEnum.height, depth=DimensionEnum.width, height=DimensionEnum.depth
        ),
        RotationType(
            width=DimensionEnum.depth, depth=DimensionEnum.height, height=DimensionEnum.width
        ),
    ]
