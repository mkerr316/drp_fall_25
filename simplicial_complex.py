# === IMPORTS ===
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import random
from typing import Set, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, QhullError


# === VECTOR SPACES ===

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


class VectorSpace(AbstractVectorSpace):
    """A concrete implementation of a real vector space, R^n.

    :param dimension: The dimension of the vector space.
    :type dimension: int
    :raises ValueError: If the dimension is not a non-negative integer.
    """

    def __init__(self, dimension: int):
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError('Dimension must be a non-negative integer.')
        self._dimension = dimension
        self._zero = None

    @property
    def field(self) -> Type[float]:
        """The underlying field, which is the real numbers (float)."""
        return float

    @property
    def dimension(self) -> int:
        """The dimension of the vector space."""
        return self._dimension

    def zero(self) -> 'Vector':
        """Returns the cached zero vector for this space.

        The zero vector is created on first access and then cached for efficiency.

        :return: The zero vector of this space.
        :rtype: Vector
        """
        if self._zero is None:
            self._zero = Vector(self, (0.0,) * self.dimension)
        return self._zero

    def vector(self, *components: float) -> 'Vector':
        """Factory method to create a vector in this space.

        :param components: The components of the vector. The number of
                           components must match the space's dimension.
        :type components: float
        :raises ValueError: If the number of components does not match the
                            space's dimension.
        :return: A new vector in this space.
        :rtype: Vector
        """
        if len(components) != self.dimension:
            raise ValueError(f"Expected {self.dimension} components, but received {len(components)}.")
        return Vector(self, tuple(float(c) for c in components))

    def __eq__(self, other) -> bool:
        return isinstance(other, VectorSpace) and self.dimension == other.dimension

    def __hash__(self) -> int:
        return hash(("VectorSpace", self.dimension))

    def __repr__(self) -> str:
        return f"R^{self.dimension}"


class Vector(AbstractVector):
    """A concrete implementation of a vector in R^n.

    The contents are stored as an immutable tuple for safety and hashability.

    :param space: The vector space to which this vector belongs.
    :type space: VectorSpace
    :param contents: A tuple of floats representing the vector's components.
    :type contents: Tuple[float, ...]
    """

    def __init__(self, space: VectorSpace, contents: Tuple[float, ...]):
        self._space = space
        self._contents = contents

    @property
    def space(self) -> VectorSpace:
        """The vector space to which this vector belongs."""
        return self._space

    @property
    def contents(self) -> Tuple[float, ...]:
        """The immutable tuple of the vector's components."""
        return self._contents

    def _check_space(self, other: 'Vector'):
        """Raises TypeError if this vector and another are not in the same space.

        :param other: The other vector to compare against.
        :type other: Vector
        :raises TypeError: If the vectors do not belong to the same space.
        """
        if self.space != other.space:
            raise TypeError('Vectors must belong to the same vector space for this operation.')

    def __add__(self, other: 'Vector') -> 'Vector':
        self._check_space(other)
        new_contents = tuple(v + w for v, w in zip(self.contents, other.contents))
        return Vector(self.space, new_contents)

    def __sub__(self, other: 'Vector') -> 'Vector':
        self._check_space(other)
        new_contents = tuple(v - w for v, w in zip(self.contents, other.contents))
        return Vector(self.space, new_contents)

    def __mul__(self, scalar: float) -> 'Vector':
        if not isinstance(scalar, (int, float)):
            raise TypeError('Scalar must be a real number.')
        if scalar == 1.0: return self
        if scalar == 0.0: return self.space.zero()
        new_contents = tuple(v * scalar for v in self.contents)
        return Vector(self.space, new_contents)

    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)

    def dot(self, other: 'Vector') -> float:
        """Computes the dot product with another vector.

        :param other: The other vector.
        :type other: Vector
        :return: The dot product of the two vectors.
        :rtype: float
        """
        self._check_space(other)
        return sum(v * w for v, w in zip(self.contents, other.contents))

    def __repr__(self) -> str:
        return f"Vector{self.contents}"


# === SETS (Used as Mixins/Markers) ===

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


# === GEOMETRIC OBJECTS ===

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


class ConvexPolytope(AbstractConvexPolytope):
    """Represents a general convex polytope in R^n.

    The minimal set of vertices is determined by computing the convex hull
    of the input vectors using SciPy.

    :param vectors: A set of vectors from which to construct the polytope.
    :type vectors: Set[Vector]
    :raises ValueError: If the set of vectors is empty.
    """

    def __init__(self, vectors: Set[Vector]):
        if not vectors:
            raise ValueError('A convex polytope requires at least one vector.')

        first_vector = next(iter(vectors))
        self._space = first_vector.space

        points_np = np.array([v.contents for v in vectors])
        try:
            hull = ConvexHull(points_np)
            vector_list = list(vectors)
            self._vertices = tuple(vector_list[i] for i in hull.vertices)
        except QhullError:
            # Fallback for degenerate cases (e.g., colinear points in 2D)
            # where hull computation fails. The set itself is the best we can do.
            self._vertices = tuple(vectors)

    @property
    def space(self) -> VectorSpace:
        """The vector space containing the polytope."""
        return self._space

    @property
    def field(self) -> Type[float]:
        """The underlying field of the vector space."""
        return float

    @property
    def vertices(self) -> Tuple[Vector, ...]:
        """The tuple of vertices that form the polytope's convex hull."""
        return self._vertices

    def __contains__(self, vector: Vector) -> bool:
        """Checks if a vector is inside the polytope.

        .. warning:: Not implemented for the general case due to complexity.

        :param vector: The vector to check.
        :type vector: Vector
        :raises NotImplementedError: This check is non-trivial and not implemented.
        """
        if not isinstance(vector, Vector) or vector.space != self.space:
            return False
        raise NotImplementedError("General polytope containment check is not yet implemented.")


class AbstractSimplex(AbstractConvexPolytope):
    """Interface for a k-simplex.

    A k-simplex is the convex hull of k+1 affinely independent vertices.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The intrinsic dimension of the simplex (k for a k-simplex)."""
        pass


class Simplex(AbstractSimplex, ConvexPolytope):
    """A k-simplex, defined by k+1 affinely independent vertices.

    This class performs its own validation to ensure the defining vertices are
    affinely independent, which is a requirement for a simplex.

    :param vertices: A set of k+1 affinely independent vertices.
    :type vertices: Set[Vector]
    :raises ValueError: If the set of vertices is empty or if the vertices
                        are not affinely independent.
    """

    def __init__(self, vertices: Set[Vector]):
        if not vertices:
            raise ValueError("A simplex requires at least one vertex.")

        self._vertices = tuple(vertices)
        self._space = self._vertices[0].space
        self._dimension = len(self._vertices) - 1

        # --- VALIDATION: Check for Affine Independence ---
        if self._dimension > 0:
            v0 = self._vertices[0]
            edge_vectors = [v - v0 for v in self._vertices[1:]]
            matrix = np.array([vec.contents for vec in edge_vectors]).T

            if np.linalg.matrix_rank(matrix) != self._dimension:
                raise ValueError(
                    f"The {len(self._vertices)} provided vertices are not affinely independent.")

    @property
    def dimension(self) -> int:
        """The intrinsic dimension 'k' of the k-simplex."""
        return self._dimension

    def __contains__(self, point: Vector) -> bool:
        """Checks if a point is inside the simplex using barycentric coordinates.

        A point 'p' is in the simplex if it can be written as a convex
        combination of the vertices v_i: p = Σ c_i * v_i, where Σ c_i = 1
        and all c_i ≥ 0.

        :param point: The point to check for containment.
        :type point: Vector
        :return: True if the point is inside the simplex, False otherwise.
        :rtype: bool
        """
        if not isinstance(point, Vector) or point.space != self.space:
            return False
        if self.dimension < 0:
            return False
        if self.dimension == 0:
            return point.contents == self.vertices[0].contents

        v0 = self.vertices[0]
        A = np.array([v.contents for v in (self.vertices[i] - v0 for i in range(1, self.dimension + 1))]).T
        b = np.array((point - v0).contents)

        try:
            coeffs = np.linalg.solve(A, b)
            c0 = 1.0 - np.sum(coeffs)
            # Check if all barycentric coordinates are non-negative within a tolerance.
            return c0 >= -1e-9 and np.all(coeffs >= -1e-9)
        except np.linalg.LinAlgError:
            # This occurs if the point is outside the affine hull of the simplex.
            return False


# === SIMPLICCIAL COMPLEXES ===

class AbstractSimplicialComplex(ABC):
    """Interface for a simplicial complex."""

    @property
    @abstractmethod
    def simplices(self) -> Set[frozenset[int]]:
        """The set of all simplices in the complex.

        Each simplex is represented by a frozenset of its vertex indices.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The highest dimension of any simplex in the complex."""
        pass


class SimplicialComplex(AbstractSimplicialComplex):
    """An abstract simplicial complex built from a random graph.

    This implementation generates a random complex using the clique complex
    model on an Erdős-Rényi random graph. A set of vertices forms a simplex
    if and only if they form a clique in the underlying graph.

    :param num_vertices: The total number of vertices in the complex.
    :type num_vertices: int
    :param p: The probability of an edge existing between any two vertices.
    :type p: float
    :raises ValueError: If 'p' is not between 0 and 1.
    """

    def __init__(self, num_vertices: int, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Probability 'p' must be between 0 and 1.")

        self._simplices: Set[frozenset[int]] = set()
        adj: dict[int, set[int]] = {v: set() for v in range(num_vertices)}

        # 1. Build the 1-skeleton (Erdős-Rényi random graph)
        for v1, v2 in itertools.combinations(range(num_vertices), 2):
            if random.random() < p:
                adj[v1].add(v2)
                adj[v2].add(v1)

        # 2. Build the clique complex from the graph.
        self._build_from_cliques(adj)

    def _build_from_cliques(self, adj: dict[int, set[int]]):
        """Populates the simplex list by finding all cliques in the graph.

        Uses a recursive backtracking algorithm (Bron-Kerbosch with pivoting) to
        find all maximal cliques. For each maximal clique found, it adds the
        clique and all of its subsets to the simplex list to ensure the
        'closed downwards' property of a valid simplicial complex is met.

        :param adj: An adjacency list representation of the graph.
        :type adj: dict[int, set[int]]
        """
        def find_cliques_recursive(potential_clique, candidates, excluded):
            if not candidates and not excluded:
                # This is a maximal clique. Add it and all its subsets (faces).
                for k in range(1, len(potential_clique) + 1):
                    for subset in itertools.combinations(potential_clique, k):
                        self._simplices.add(frozenset(subset))
                return

            pivot = next(iter(candidates | excluded), None)
            if pivot is None: return

            for v in list(candidates - adj[pivot]):
                find_cliques_recursive(potential_clique + [v], candidates.intersection(adj[v]),
                                       excluded.intersection(adj[v]))
                candidates.remove(v)
                excluded.add(v)

        find_cliques_recursive([], set(adj.keys()), set())

    @property
    def simplices(self) -> Set[frozenset[int]]:
        """Returns the set of all simplices in the complex."""
        return self._simplices

    @property
    def dimension(self) -> int:
        """Returns the dimension of the complex.

        The dimension of an empty complex is defined as -1.
        """
        if not self.simplices:
            return -1
        return max(len(s) for s in self.simplices) - 1

    def __repr__(self) -> str:
        num_simplices = len(self.simplices)
        num_vertices = len([s for s in self.simplices if len(s) == 1])
        return f"SimplicialComplex(vertices={num_vertices}, simplices={num_simplices}, dim={self.dimension})"


# === ANALYSIS FUNCTIONS ===

def compute_euler_characteristic(simplicial_complex: SimplicialComplex) -> int:
    """Computes the Euler Characteristic (χ) for a given simplicial complex.

    The Euler characteristic is calculated as the alternating sum of the number
    of simplices of each dimension: χ = k₀ - k₁ + k₂ - k₃ + ..., where kₙ
    is the number of n-simplices.

    :param simplicial_complex: An instance of the SimplicialComplex class.
    :type simplicial_complex: SimplicialComplex
    :return: The integer value of the Euler characteristic.
    :rtype: int
    """
    if not simplicial_complex.simplices:
        return 0

    counts = defaultdict(int)
    for simplex in simplicial_complex.simplices:
        dim = len(simplex) - 1
        if dim >= 0:
            counts[dim] += 1

    max_dim = max(counts.keys())
    return sum(((-1) ** dim) * counts.get(dim, 0) for dim in range(max_dim + 1))


def generate_complexes(num_vertices: int, p: float, num_complexes: int) -> list[int]:
    """Generates multiple random complexes and returns their Euler characteristics.

    :param num_vertices: Number of vertices for each complex.
    :type num_vertices: int
    :param p: Edge probability for each complex.
    :type p: float
    :param num_complexes: The number of complexes to generate.
    :type num_complexes: int
    :return: A list of Euler characteristic values.
    :rtype: list[int]
    """
    return [compute_euler_characteristic(SimplicialComplex(num_vertices, p)) for _ in range(num_complexes)]


# === MAIN EXECUTION ===

def main():
    """Run the simulation and display the results.

    This function sets parameters, generates random simplicial complexes,
    computes their Euler characteristics, and then analyzes and visualizes
    the distribution of these values.
    """
    # --- Parameters ---
    NUM_VERTICES = 10
    NUM_COMPLEXES = 100
    PROBABILITY = 0.4

    # --- Simulation ---
    print(f"Generating {NUM_COMPLEXES} complexes with n={NUM_VERTICES} vertices and p={PROBABILITY}...")
    chi_values = generate_complexes(NUM_VERTICES, PROBABILITY, NUM_COMPLEXES)
    print("Simulation complete.")

    # --- Analysis ---
    avg_chi = np.mean(chi_values)
    median_chi = np.median(chi_values)
    std_chi = np.std(chi_values)
    print(f"\n--- Analysis of Results ---")
    print(f"Average Euler Characteristic (χ): {avg_chi:.2f}")
    print(f"Median Euler Characteristic (χ):  {median_chi:.2f}")
    print(f"Standard Deviation of χ:        {std_chi:.2f}")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Euler Characteristic (χ) Distribution for {NUM_COMPLEXES} Random Complexes\n'
                 f'(n={NUM_VERTICES}, p={PROBABILITY})', fontsize=16)

    ax1.hist(chi_values, bins=range(min(chi_values), max(chi_values) + 2), align='left', edgecolor='black',
             color='skyblue')
    ax1.set_title('Frequency Distribution')
    ax1.set_xlabel('Euler Characteristic (χ)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(avg_chi, color='r', linestyle='dashed', linewidth=2, label=f'Average χ = {avg_chi:.2f}')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.boxplot(chi_values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax2.set_title('Box Plot Summary')
    ax2.set_xlabel('Euler Characteristic (χ)')
    ax2.set_yticklabels([''])
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

    # --- Discussion of Findings ---
    print("\n--- Discussion of Findings ---")
    print(f"""
The **Euler characteristic (χ)** is a topological invariant, a number that describes a topological space's structure
regardless of how it is bent or deformed. For a simplicial complex, it is the alternating sum of the number of
simplices of each dimension (χ = #vertices - #edges + #faces - ...).

**Key Observations from the Simulation:**

* **Distribution:** The histogram shows that the χ values for random complexes generated with these parameters
    (n={NUM_VERTICES}, p={PROBABILITY}) are not uniform. They cluster around a central value, which in this case is negative. The box plot
    confirms this central tendency and shows the interquartile range.

* **Interpretation of the Average Value:**
    - A completely disconnected set of {NUM_VERTICES} vertices would have χ = {NUM_VERTICES}.
    - A single, contractible ("blob-like") complex with no holes has χ = 1.
    - Our average of **{avg_chi:.2f}** is significantly different from these simple cases. The negative value is particularly
      informative. It suggests that, on average, the number of 1-simplices (edges) is large enough to overwhelm
      the number of 0-simplices (vertices).

* **The Role of 'Holes':** The prevalence of negative χ values strongly indicates the formation of
    **1-dimensional holes (cycles or loops)**. At p={PROBABILITY}, the graph is dense enough to have many cycles (e.g., C₃, C₄),
    but not necessarily dense enough to "fill in" all these cycles with 2-simplices (triangles) or higher-dimensional
    simplices. Each unfilled cycle tends to reduce the Euler characteristic.

* **Conclusion:** This simulation acts as a form of computational topology. It reveals the "typical" shape, or
    homological signature, produced by the Erdős-Rényi random graph process for a given `n` and `p`.
    Changing `p` would drastically alter this distribution:
    - A very low `p` would result in a mostly disconnected graph, with χ values clustering near n={NUM_VERTICES}.
    - A very high `p` would result in a highly connected graph with many filled-in cliques, likely pushing the
      average χ towards 1.
""")


if __name__ == "__main__":
    main()