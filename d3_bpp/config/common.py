"""Utility functions and classes for the 3d-bpp package."""
from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, ParamSpec, TypeVar

V = TypeVar("V", int, float)


@dataclass(frozen=True)
class MinMaxTuple(Generic[V]):
    """Helper class to define a pair of min/max values."""

    min: V
    """Minimum value."""
    max: V
    """Maximum value."""

    def __post_init__(self) -> None:
        """Check that the minimum is smaller than the maximum."""
        if self.min > self.max:
            raise ValueError(
                f"Min value of the tuple {self.min} cannot be greater than the max {self.max}."
            )

    @classmethod
    def from_iterable(cls, it_: Iterable[V]) -> MinMaxTuple[V]:
        """Create a MinMaxTuple from an iterable."""
        min_ = None
        max_ = None
        for i, v in enumerate(it_):
            if i == 0:
                min_ = v
            if i == 1:
                max_ = v
            if i > 1:
                raise ValueError(
                    f"Cannot create a MinMaxTuple from an iterable with more than 2 elements: {it_}"
                )
        if min_ is None or max_ is None:
            raise ValueError(
                f"Cannot create a MinMaxTuple from an iterable with less than 2 elements: {it_}"
            )
        return cls(min_, max_)


P = ParamSpec("P")


class Singleton(ABCMeta):
    """Singleton metaclass."""

    _instances: dict[type, Singleton] = {}
    """Dictionary of instances of the class."""

    def __call__(cls, *args: P.args, **kwargs: P.kwargs) -> Singleton:
        """Return the instance of the class."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
