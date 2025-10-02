# === VECTOR SPACES ===
from abc import ABC, abstractmethod
from typing import Type, Set, Tuple


class AbstractVectorSpace(ABC):
    """Defines the abstract interface for a vector space."""

    @property
    @abstractmethod
    def field(self) -> Type:
        """The underlying field for the vector space (e.g., float)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of the vector space."""
        pass

    @abstractmethod
    def zero(self) -> 'AbstractVector':
        """Returns the zero vector, the additive identity of the space.

        :return: The zero vector.
        :rtype: AbstractVector
        """
        pass

    def __contains__(self, vector: 'AbstractVector') -> bool:
        """Checks if a vector is a member of this vector space.

        :param vector: The vector to check.
        :type vector: AbstractVector
        :return: True if the vector belongs to this space, False otherwise.
        :rtype: bool
        """
        return isinstance(vector, AbstractVector) and vector.space == self


class AbstractVector(ABC):
    """Defines the abstract interface for a vector in a vector space."""

    @property
    @abstractmethod
    def space(self) -> AbstractVectorSpace:
        """The vector space to which this vector belongs."""
        pass

    @abstractmethod
    def __add__(self, other: 'AbstractVector') -> 'AbstractVector': pass

    @abstractmethod
    def __sub__(self, other: 'AbstractVector') -> 'AbstractVector': pass

    @abstractmethod
    def __mul__(self, scalar) -> 'AbstractVector': pass

    @abstractmethod
    def __rmul__(self, scalar) -> 'AbstractVector': pass

# === AFFINE SPACES ===

class AbstractAffineSpace(ABC):
    """Defines the abstract interface for an affine space."""
    pass

class AbstractAffinePoint(ABC):
    """Defines the abstract interface for a point in an affine space."""
    pass

# === SETS ===
class BoundedSet(ABC):
    """Mixin class to mark a set as bounded."""
    pass


class ClosedSet(ABC):
    """Mixin class to mark a set as closed."""
    pass

class CompactSet(ClosedSet, BoundedSet):
    """Mixin class to mark a set as compact (closed and bounded)."""
    pass


class ConvexSet(ABC):
    """Mixin class to mark a set as convex."""
    pass

# === SIMPLICIAL COMPLEXES ===
class AbstractSimplicialComplex(ABC):
    """Interface for a simplicial complex."""

    @property
    @abstractmethod
    def simplices(self) -> Set[frozenset[int]]: pass

    @property
    @abstractmethod
    def dimension(self) -> int: pass

class AbstractConvexPolytope(CompactSet, ConvexSet):
    """Interface for a convex polytope.

    A convex polytope is defined as the convex hull of a finite set of points
    (its vertices). It is a compact and convex set.
    """

    @property
    @abstractmethod
    def vertices(self) -> Tuple[AbstractVector, ...]:
        """The minimal set of vertices defining the polytope."""
        pass

    @abstractmethod
    def __contains__(self, other: AbstractVector) -> bool: pass

class AbstractSimplex(AbstractConvexPolytope):
    """Interface for a k-simplex.

    A k-simplex is the convex hull of k+1 affinely independent vertices.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The intrinsic dimension of the simplex (k for a k-simplex)."""
        pass