"""Module containing container measures."""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

from .common import Singleton

ContainerTypes: dict[ContainerTypesEnum, type[ContainerMeasures]] = {}
"""Dictionary containing the available container types."""


class ContainerTypesEnum(str, Enum):
    """Container types enumeration."""

    twenty_foot = "twenty_foot"
    """20ft. Container."""
    forty_foot = "forty_foot"
    """40ft. Container."""
    forty_foot_high_cube = "forty_foot_high_cube"
    """40ft. High Cube Container."""


@dataclass(frozen=True, order=True)
class ContainerMeasures(ABC, metaclass=Singleton):
    """Container measures (mm,kg)."""

    type_: ContainerTypesEnum = field(init=False)
    """Container type."""

    inner_width: int = field(init=False)
    """Container inner width (mm)."""
    inner_depth: int = field(init=False)
    """Container inner depth (mm)."""
    inner_height: int = field(init=False)
    """Container inner height (mm)."""

    door_width: int = field(init=False)
    """Container door width (mm)."""
    door_height: int = field(init=False)
    """Container door height (mm)."""

    empty_weight: int = field(init=False)
    """Empty weight of the container (kg)."""
    gross_mass: int = field(init=False)
    """Maximum gross mass of the container (kg)."""
    load_weight: int = field(init=False)
    """Maximum loaded container weight (kg)."""

    def __post_init__(self) -> None:
        """Compute the load weight."""
        if self.gross_mass < self.empty_weight:
            raise ValueError("Gross mass cannot be smaller than empty weight.")
        if self.door_height > self.inner_height:
            raise ValueError("Door height cannot be greater than inner height.")
        if self.door_width > self.inner_width:
            raise ValueError("Door width cannot be greater than inner width.")
        object.__setattr__(self, "load_weight", self.gross_mass - self.empty_weight)

    def __init_subclass__(cls) -> None:
        """Add the class to the ContainerTypes dictionary."""
        if cls.type_ in ContainerTypes:
            raise ValueError(f"Container type {cls.type_} already exists.")
        ContainerTypes[cls.type_] = cls


@dataclass(frozen=True, order=True)
class Container_20ft_Measures(ContainerMeasures):
    """Standard 20ft. Container measures (mm,kg)."""

    type_: ContainerTypesEnum = ContainerTypesEnum.twenty_foot

    inner_width: int = 2310
    """Container inner width (mm)."""
    inner_depth: int = 5860
    """Container inner depth (mm)."""
    inner_height: int = 2360
    """Container inner height (mm)."""

    door_width: int = 2280
    """Container door width (mm)."""
    door_height: int = 2270
    """Container door height (mm)."""

    empty_weight: int = 2650
    """Empty weight of the container (kg)."""
    gross_mass: int = 27980
    """Maximum gross mass of the container (kg)."""


@dataclass(frozen=True, order=True)
class Container_40ft_Measures(ContainerMeasures):
    """Standard 40ft. Container dimensions (mm,kg)."""

    type_: ContainerTypesEnum = ContainerTypesEnum.forty_foot

    inner_width: int = 2310
    """Container inner width (mm)."""
    inner_depth: int = 12010
    """Container inner depth (mm)."""
    inner_height: int = 2360
    """Container inner height (mm)."""

    door_width: int = 2280
    """Container door width (mm)."""
    door_height: int = 2270
    """Container door height (mm)."""

    empty_weight: int = 3740
    """Empty weight of the container (kg)."""
    gross_mass: int = 36850
    """Maximum gross mass of the container (kg)."""


@dataclass(frozen=True, order=True)
class Container_40ft_High_Cube_Measures(ContainerMeasures):
    """Standard 40ft. High Cube Container dimensions (mm,kg)."""

    type_: ContainerTypesEnum = ContainerTypesEnum.forty_foot_high_cube

    inner_width: int = 2340
    """Container inner width (mm)."""
    inner_depth: int = 12020
    """Container inner depth (mm)."""
    inner_height: int = 2680
    """Container inner height (mm)."""

    door_width: int = 2290
    """Container door width (mm)."""
    door_height: int = 2570
    """Container door height (mm)."""

    empty_weight: int = 4150
    """Empty weight of the container (kg)."""
    gross_mass: int = 36600
    """Maximum gross mass of the container (kg)."""
