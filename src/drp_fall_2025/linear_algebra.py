from typing import Tuple, TypeVar
from .fields import Field
from .interfaces import AbstractVectorSpace, AbstractVector

# Generic type for field elements (e.g., int for PrimeField, float for R)
FieldElement = TypeVar('FieldElement')


class VectorSpace(AbstractVectorSpace):
    """A concrete implementation of a vector space over a given field."""

    def __init__(self, dimension: int, field: Field):
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError("Dimension must be a non-negative integer.")
        self._dimension = dimension
        self._field = field
        self._zero = None

    @property
    def field(self) -> Field:
        return self._field

    @property
    def dimension(self) -> int:
        return self._dimension

    def zero(self) -> 'Vector':
        if self._zero is None:
            # Get the additive identity from the field object
            zero_element = self.field.zero
            self._zero = Vector(self, (zero_element,) * self.dimension)
        return self._zero

    def vector(self, *components: FieldElement) -> 'Vector':
        """Creates a vector from components belonging to the space's field."""
        if len(components) != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} components, but got {len(components)}."
            )
        # It's good practice to ensure components are valid field elements,
        # but for performance, we can rely on the user providing correct types.
        return Vector(self, tuple(components))


class Vector(AbstractVector):
    """A concrete implementation of a vector in a generic vector space."""

    def __init__(self, space: VectorSpace, contents: Tuple[FieldElement, ...]):
        self._space = space
        self._contents = contents

    @property
    def space(self) -> VectorSpace:
        return self._space

    @property
    def contents(self) -> Tuple[FieldElement, ...]:
        return self._contents

    def _check_space(self, other: 'Vector'):
        if self.space != other.space:
            raise TypeError('Vectors must belong to the same space for this operation.')

    # Your arithmetic operations are already correctly implemented!
    # They properly delegate all calculations to the field object.
    # No changes are needed for __add__, __sub__, __mul__, __rmul__, or dot.
    # ... (keep existing methods as they are) ...

    def __add__(self, other: 'Vector') -> 'Vector':
        self._check_space(other)
        field = self.space.field
        new_contents = tuple(
            field.add(v, w) for v, w in zip(self.contents, other.contents)
        )
        return Vector(self.space, new_contents)

    def __sub__(self, other: 'Vector') -> 'Vector':
        self._check_space(other)
        field = self.space.field
        new_contents = tuple(
            field.sub(v, w) for v, w in zip(self.contents, other.contents)
        )
        return Vector(self.space, new_contents)

    def __mul__(self, scalar: FieldElement) -> 'Vector':
        field = self.space.field
        new_contents = tuple(field.mul(c, scalar) for c in self.contents)
        return Vector(self.space, new_contents)

    def __rmul__(self, scalar: FieldElement) -> 'Vector':
        return self.__mul__(scalar)

    def dot(self, other: 'Vector') -> FieldElement:
        self._check_space(other)
        field = self.space.field
        total = field.zero
        for v, w in zip(self.contents, other.contents):
            product = field.mul(v, w)
            total = field.add(total, product)
        return total

    def __repr__(self) -> str:
        return f"Vector{self.contents}"