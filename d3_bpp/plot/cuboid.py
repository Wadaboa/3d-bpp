"""Module to manage 3D cuboid information."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True, order=True)
class Dimension:
    """Helper class to define object dimensions."""

    width: int
    """Width of the object (mm)."""
    depth: int
    """Depth of the object (mm)."""
    height: int
    """Height of the object (mm)."""
    weight: int = 0
    """Weight of the object (kg)."""
    area: int = field(init=False)
    """Area of the object (mm^2)."""
    volume: int = field(init=False)
    """Volume of the object (mm^3)."""

    def __post_init__(self, width: int, depth: int, height: int, weight: int) -> None:
        """Initialize area and volume attributes of the Dimension."""
        object.__setattr__(self, "area", width * depth)
        object.__setattr__(self, "volume", width * depth * height)


@dataclass(frozen=True, order=True)
class Coordinate:
    """Helper class to define a pair/triplet of coordinates.

    The coordinates are stored in the corresponding axis, i.e. x, y, z.
    """

    x: int
    """X coordinate."""
    y: int
    """Y coordinate."""
    z: int = 0
    """Z coordinate."""

    def to_tuple(self) -> tuple[int, int, int]:
        """Convert coordinates to a tuple."""
        return self.x, self.y, self.z

    def __getitem__(self, key: int) -> int:
        """Return the coordinate in the specified axis."""
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            raise IndexError("Index out of range.")

    def __iter__(self) -> Iterable[int]:
        """Iterate over the coordinates."""
        return iter(self.to_tuple())


@dataclass(frozen=True, order=True)
class Cuboid_Face:
    """Helper class to define a cuboid face.

    The face is defined by its four corners in the following order:
    1. bottom left
    2. bottom right
    3. top right
    4. top left
    """

    bottom_left: Coordinate
    """Bottom left corner of the face."""
    bottom_right: Coordinate
    """Bottom right corner of the face."""
    top_right: Coordinate
    """Top right corner of the face."""
    top_left: Coordinate
    """Top left corner of the face."""

    def to_tuple(self) -> tuple[Coordinate, Coordinate, Coordinate, Coordinate]:
        """Convert face to a tuple."""
        return self.bottom_left, self.bottom_right, self.top_right, self.top_left


@dataclass(frozen=True, order=True)
class Cuboid:
    """Helper class to define the set of vertices identifying a cuboid."""

    blb: Coordinate
    """Bottom left back vertex in space of the cuboid."""
    dims: Dimension
    """Dimensions of the cuboid."""

    @cached_property
    def blf(self) -> Coordinate:
        """Bottom left front vertex in space of the cuboid."""
        return Coordinate(self.blb.x + self.dims.width, self.blb.y, self.blb.z)

    @cached_property
    def brb(self) -> Coordinate:
        """Bottom right back vertex in space of the cuboid."""
        return Coordinate(self.blb.x, self.blb.y + self.dims.depth, self.blb.z)

    @cached_property
    def brf(self) -> Coordinate:
        """Bottom right front vertex in space of the cuboid."""
        return Coordinate(self.blb.x + self.dims.width, self.blb.y + self.dims.depth, self.blb.z)

    @cached_property
    def tlb(self) -> Coordinate:
        """Top left back vertex in space of the cuboid."""
        return Coordinate(self.blb.x, self.blb.y, self.blb.z + self.dims.height)

    @cached_property
    def tlf(self) -> Coordinate:
        """Top left front vertex in space of the cuboid."""
        return Coordinate(self.blb.x + self.dims.width, self.blb.y, self.blb.z + self.dims.height)

    @cached_property
    def trb(self) -> Coordinate:
        """Top right back vertex in space of the cuboid."""
        return Coordinate(self.blb.x, self.blb.y + self.dims.depth, self.blb.z + self.dims.height)

    @cached_property
    def trf(self) -> Coordinate:
        """Top right front vertex in space of the cuboid."""
        return Coordinate(
            self.blb.x + self.dims.width,
            self.blb.y + self.dims.depth,
            self.blb.z + self.dims.height,
        )

    @cached_property
    def vertices(self) -> tuple[Coordinate, ...]:
        """Vertices of the cuboid.

        The order is:
        1. `bottom-left-back`
        2. `bottom-left-front`
        3. `bottom-right-back`
        4. `bottom-right-front`
        5. `top-left-back`
        6. `top-left-front`
        7. `top-right-back`
        8. `top-right-front`
        """
        return (self.blb, self.blf, self.brb, self.brf, self.tlb, self.tlf, self.trb, self.trf)

    @cached_property
    def center(self) -> Coordinate:
        """Return the coordinates of the center of the cuboid."""
        return Coordinate(
            self.blb.x + self.dims.width // 2,
            self.blb.y + self.dims.depth // 2,
            self.blb.z + self.dims.height // 2,
        )

    @cached_property
    def xs(self) -> tuple[int, ...]:
        """Return a tuple containing all the x-values of the vertices of the cuboid.

        The order is the same as the order of the vertices.
        """
        return tuple(vertex.x for vertex in self.vertices)

    @cached_property
    def ys(self) -> tuple[int, ...]:
        """Return a tuple containing all the y-values of the vertices of the cuboid.

        The order is the same as the order of the vertices.
        """
        return tuple(vertex.y for vertex in self.vertices)

    @cached_property
    def zs(self) -> tuple[int, ...]:
        """Return a tuple containing all the z-values of the vertices of the cuboid.

        The order is the same as the order of the vertices.
        """
        return tuple(vertex.z for vertex in self.vertices)

    @cached_property
    def bottom_face(self) -> Cuboid_Face:
        """Return the bottom face of the cuboid."""
        return Cuboid_Face(self.blb, self.blf, self.brf, self.brb)

    @cached_property
    def top_face(self) -> Cuboid_Face:
        """Return the top face of the cuboid."""
        return Cuboid_Face(self.tlb, self.tlf, self.trf, self.trb)

    @cached_property
    def back_face(self) -> Cuboid_Face:
        """Return the back face of the cuboid."""
        return Cuboid_Face(self.blb, self.brb, self.trb, self.tlb)

    @cached_property
    def front_face(self) -> Cuboid_Face:
        """Return the front face of the cuboid."""
        return Cuboid_Face(self.blf, self.brf, self.trf, self.tlf)

    @cached_property
    def left_face(self) -> Cuboid_Face:
        """Return the left face of the cuboid."""
        return Cuboid_Face(self.blb, self.blf, self.tlf, self.tlb)

    @cached_property
    def right_face(self) -> Cuboid_Face:
        """Return the right face of the cuboid."""
        return Cuboid_Face(self.brb, self.brf, self.trf, self.trb)

    @cached_property
    def faces(self) -> tuple[Cuboid_Face, ...]:
        """Faces of the cuboid.

        The order is:
        1. `bottom`
        2. `top`
        3. `back`
        4. `front`
        5. `left`
        6. `right`
        """
        return (
            self.bottom_face,
            self.top_face,
            self.back_face,
            self.front_face,
            self.left_face,
            self.right_face,
        )
