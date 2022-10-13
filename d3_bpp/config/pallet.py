"""Module containing pallet measures."""
from __future__ import annotations

from abc import ABC
from dataclasses import InitVar, dataclass, field
from enum import Enum

from .common import Singleton
from .container import ContainerMeasures

PalletTypes: dict[str, type[PalletMeasures]] = {}
"""Dictionary containing the available pallet types."""


class PalletTypesEnum(str, Enum):
    """Pallet types enumeration."""

    eur_1 = "eur_1"
    """EUR 1 pallet."""
    eur_3 = "eur_3"
    """EUR 3 pallet."""
    eur_6 = "eur_6"
    """EUR 6 pallet."""


@dataclass(frozen=True, order=True)
class PalletMeasures(ABC, metaclass=Singleton):
    """Pallet measures (mm,kg)."""

    container: InitVar[ContainerMeasures]
    """Container type in which the pallet will be placed."""

    type_: PalletTypesEnum = field(init=False)
    """Pallet type."""

    base_height: int = field(init=False)
    """Pallet base height (mm)."""
    base_width: int = field(init=False)
    """Pallet base width (mm)."""
    base_depth: int = field(init=False)
    """Pallet base depth (mm)."""

    empty_weight: int = field(init=False)
    """Empty weight of the pallet (kg)."""
    gross_mass: int = field(init=False)
    """Maximum gross mass of the pallet (kg)."""
    load_weight: int = field(init=False)
    """Maximum loaded pallet weight (kg)."""

    load_height: int = field(init=False)
    """Maximum loaded pallet height (mm)."""

    def __post_init__(self, container: ContainerMeasures) -> None:
        """Compute the load weight."""
        object.__setattr__(self, "load_weight", self.gross_mass - self.empty_weight)
        object.__setattr__(self, "load_height", container.door_height - self.base_height)

    def __init_subclass__(cls) -> None:
        """Add the class to the PalletTypes dictionary."""
        if cls.type_ in PalletTypes:
            raise ValueError(f"Pallet type {cls.type_} already exists.")
        PalletTypes[cls.type_] = cls


@dataclass(frozen=True, order=True)
class Pallet_EUR1_Dimension(PalletMeasures):
    """Pallet of type EUR1 supported dimensions."""

    type_: PalletTypesEnum = PalletTypesEnum.eur_1
    """Pallet type."""

    base_height: int = 144
    """Pallet base height (mm)."""
    base_width: int = 1200
    """Pallet base width (mm)."""
    base_depth: int = 800
    """Pallet base depth (mm)."""

    empty_weight: int = 25
    """Empty weight of the pallet (kg)."""
    gross_mass: int = 2515
    """Maximum gross mass of the pallet (kg)."""


@dataclass(frozen=True, order=True)
class Pallet_EUR3_Dimension(PalletMeasures):
    """Pallet of type EUR3 supported dimensions."""

    type_: PalletTypesEnum = PalletTypesEnum.eur_3
    """Pallet type."""

    base_height: int = 144
    """Pallet base height (mm)."""
    base_width: int = 1200
    """Pallet base width (mm)."""
    base_depth: int = 1000
    """Pallet base depth (mm)."""

    empty_weight: int = 29
    """Empty weight of the pallet (kg)."""
    gross_mass: int = 1949
    """Maximum gross mass of the pallet (kg)."""


@dataclass(frozen=True, order=True)
class Pallet_EUR6_Dimension(PalletMeasures):
    """Pallet of type EUR6 supported dimensions."""

    type_: PalletTypesEnum = PalletTypesEnum.eur_6
    """Pallet type."""

    base_height: int = 144
    """Pallet base height (mm)."""
    base_width: int = 800
    """Pallet base width (mm)."""
    base_depth: int = 600
    """Pallet base depth (mm)."""

    empty_weight: int = 10
    """Empty weight of the pallet (kg)."""
    gross_mass: int = 510
    """Maximum gross mass of the pallet (kg)."""
