"""Module to represent a cuboid."""
from __future__ import annotations

from abc import ABC, ABCMeta
from collections.abc import Collection, Iterable, Iterator
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import NewType


class AxisEnum(str, Enum):
    """Axis enumeration."""

    x = "x"
    """X axis."""
    y = "y"
    """Y axis."""
    z = "z"
    """Z axis."""

    def get_dimension(self) -> DimensionEnum:
        """Return the corresponding dimension."""
        return AxisDimensionAssociation[self]


class DimensionEnum(str, Enum):
    """Dimension enumeration."""

    width = "width"
    """Width dimension."""
    depth = "depth"
    """Depth dimension."""
    height = "height"
    """Height dimension."""

    def get_axis(self) -> AxisEnum:
        """Return the corresponding axis."""
        return DimensionAxisAssociation[self]


AxisDimensionAssociation: dict[AxisEnum, DimensionEnum] = {
    AxisEnum.x: DimensionEnum.width,
    AxisEnum.y: DimensionEnum.depth,
    AxisEnum.z: DimensionEnum.height,
}
"""Dictionary containing the association between axis and dimension."""

DimensionAxisAssociation: dict[DimensionEnum, AxisEnum] = {
    key: value for value, key in AxisDimensionAssociation.items()
}
"""Dictionary containing the association between dimension and axis."""


Dimension = NewType("Dimension", int)
"""Spatial dimension associated with a Dimension type (mm)."""
Coordinate = NewType("Coordinate", int)
"""Spatial coordinate associated with an axis."""


@dataclass(kw_only=True, frozen=True, order=True)
class CuboidDimension(Iterable[Dimension]):
    """Helper class to define cuboid dimensions."""

    width: Dimension
    """Width of the cuboid (mm)."""
    depth: Dimension
    """Depth of the cuboid (mm)."""
    height: Dimension
    """Height of the cuboid (mm)."""
    area: int = field(init=False)
    """Area of the base of the cuboid (mm^2)."""
    volume: int = field(init=False)
    """Volume of the cuboid (mm^3)."""

    def __post_init__(self) -> None:
        """Initialize area and volume attributes of the Dimension."""
        object.__setattr__(self, "area", self.width * self.depth)
        object.__setattr__(self, "volume", self.area * self.height)

    def __getitem__(self, key: DimensionEnum | AxisEnum) -> Dimension:
        """Return the dimension of the specified type."""
        if isinstance(key, DimensionEnum):
            return self.get_dimension(key)
        elif isinstance(key, AxisEnum):
            return self.get_axis(key)
        else:
            raise ValueError(f"Key {key} not recognized.")

    def __iter__(self) -> Iterator[Dimension]:
        """Return an iterator over the dimensions."""
        yield from (self.get_dimension(dimension=dimension) for dimension in DimensionEnum)

    def get_dimension(self, dimension: DimensionEnum) -> Dimension:
        """Return the value of the specified dimension."""
        if dimension == DimensionEnum.width:
            return self.width
        elif dimension == DimensionEnum.depth:
            return self.depth
        elif dimension == DimensionEnum.height:
            return self.height
        else:
            raise IndexError(f"Dimension type {dimension} not recognized.")

    def get_axis(self, axis: AxisEnum) -> Dimension:
        """Return the value of the specified axis."""
        return self.get_dimension(axis.get_dimension())
