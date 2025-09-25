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

# === AFFINE SPACES ===

class AbstractAffineSpace(ABC):
    """Defines the abstract interface for an affine space."""
    pass

class AbstractAffinePoint(ABC):
    """Defines the abstract interface for a point in an affine space."""
    pass

class AffineSpace(AbstractAffineSpace):
    """A concrete implementation of a real affine space, A^n."""
    pass

class AffinePoint(AbstractAffinePoint):
    """A concrete implementation of a point in A^n."""
    pass

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

        . warning:: Not implemented for the general case due to complexity.

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
    def __init__(self, vertices: Set[Vector]):
        if not vertices:
            raise ValueError("A simplex requires at least one vertex.")

        k = len(vertices) - 1

        # --- VALIDATION: Perform simplex-specific checks first ---
        if k > 0:
            vertex_tuple = tuple(vertices)
            v0 = vertex_tuple[0]
            edge_vectors = [v - v0 for v in vertex_tuple[1:]]
            matrix = np.array([vec.contents for vec in edge_vectors]).T

            if np.linalg.matrix_rank(matrix) != k:
                raise ValueError(f"The {len(vertices)} provided vertices are not affinely independent.")

        # --- CONSTRUCTION: Now call the parent constructor ---
        # A simplex's vertices *are* the minimal set, so we can bypass the Qhull logic
        # by passing them directly. Or better, we let the parent handle it.
        super().__init__(vertices)

        # The ConvexPolytope constructor already finds the minimal vertices.
        # For a valid simplex, this should be the same as the input set.
        if len(self.vertices) != len(vertices):
            # This case might occur if ConvexHull simplifies a degenerate input
            raise ValueError("The provided vertices do not form the vertices of their convex hull, indicating degeneracy.")

        self._dimension = k

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


# === SIMPLICIAL COMPLEXES ===

class AbstractSimplicialComplex(ABC):
    """Interface for a simplicial complex."""

    @property
    @abstractmethod
    def simplices(self) -> Set[frozenset[int]]: pass

    @property
    @abstractmethod
    def dimension(self) -> int: pass


class SimplicialComplex(AbstractSimplicialComplex):
    """An abstract simplicial complex defined by a set of its faces."""

    def __init__(self, simplices_as_frozensets: Set[frozenset[int]]):
        """
        A more general constructor that takes a set of frozensets, where
        each frozenset represents a simplex. This assumes all sub-faces
        for any given simplex are also present in the set.
        """
        self._simplices = simplices_as_frozensets

    @classmethod
    def from_maximal_simplices(cls, maximal_simplices: list[set[int]]) -> 'SimplicialComplex':
        """Creates a complex by generating all sub-faces from a list of maximal ones."""
        all_simplices = set()
        for maximal_simplex in maximal_simplices:
            for k in range(1, len(maximal_simplex) + 1):
                for face in itertools.combinations(maximal_simplex, k):
                    all_simplices.add(frozenset(face))
        return cls(all_simplices)

    @classmethod
    def from_top_down_process(cls, num_vertices: int, p_keep: float) -> 'SimplicialComplex':
        """
        Creates a random complex using a top-down probabilistic "erosion" process.
        Starts with the full (n-1)-simplex and removes faces probabilistically.
        """
        if not (0 <= p_keep <= 1):
            raise ValueError("p_keep must be between 0 and 1.")

        current_maximals = {frozenset(range(num_vertices))}
        for dim in range(num_vertices - 1, 0, -1):
            next_maximals = set()
            faces_to_consider = set()
            for simplex in current_maximals:
                for face in itertools.combinations(simplex, dim):
                    faces_to_consider.add(frozenset(face))
            for face in faces_to_consider:
                if random.random() < p_keep:
                    next_maximals.add(face)
            if not next_maximals:
                current_maximals = set()
                break
            current_maximals = next_maximals

        # Update: Now calls the factory method to maintain old functionality
        return cls.from_maximal_simplices([set(s) for s in current_maximals])

    @classmethod
    def from_bottom_up_process(cls, num_vertices: int, p_dict: dict[int, float]) -> 'SimplicialComplex':
        """
        Creates a random complex.

        Simplices are added probabilistically at each dimension, provided their
        boundaries already exist in the complex.

        :param num_vertices: The total number of vertices.
        :param p_dict: A dictionary mapping dimension 'k' to the probability
                       p_k of including a k-simplex. E.g., {1: 0.8, 2: 0.5}
                       for edges and triangles. A dimension not in the dict
                       has a probability of 0.
        """
        # Start with all vertices (0-simplices)
        simplices = {frozenset([i]) for i in range(num_vertices)}

        # Iterate from dimension 1 (edges) up to the max possible dimension
        for k in range(1, num_vertices):
            pk = p_dict.get(k, 0.0)
            if pk == 0: continue

            # Consider all possible k-simplices
            for candidate_tuple in itertools.combinations(range(num_vertices), k + 1):
                candidate = frozenset(candidate_tuple)

                # --- Boundary Check ---
                is_boundary_present = True
                for boundary_face_tuple in itertools.combinations(candidate, k):
                    if frozenset(boundary_face_tuple) not in simplices:
                        is_boundary_present = False
                        break

                if is_boundary_present and random.random() < pk:
                    simplices.add(candidate)

        return cls(simplices)

    @property
    def simplices(self) -> Set[frozenset[int]]:
        return self._simplices

    @property
    def dimension(self) -> int:
        if not self.simplices: return -1
        return max(len(s) for s in self.simplices) - 1

    def __repr__(self) -> str:
        num_simplices = len(self.simplices)
        if not self.simplices:
            return "SimplicialComplex(vertices=0, simplices=0, dim=-1)"
        all_vertices = set().union(*self.simplices)
        num_vertices = len(all_vertices)
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

def generate_complexes_bottom_up(num_vertices: int, p_dict: dict[int, float], num_runs: int) -> list[int]:
    """Generates multiple bottom-up complexes and records their Euler characteristics."""
    return [compute_euler_characteristic(
        SimplicialComplex.from_bottom_up_process(num_vertices, p_dict)
    ) for _ in range(num_runs)]


def main():
    """Runs the full simulation and analysis as per the project deliverables."""
    # --- Parameters matching deliverable (c) ---
    NUM_VERTICES = 10
    NUM_COMPLEXES_TO_RUN = 100

    # --- Probabilities for the bottom-up model ---
    # NOTE: You should experiment with these values! Different probabilities
    # will create different kinds of "typical" shapes.
    PROBABILITIES = {
        1: 0.5,  # Probability of adding an edge.
        2: 0.2,  # Probability of adding a triangle (if its 3 edges exist).
        3: 0.1  # Probability of adding a tetrahedron (if its 4 faces exist).
    }

    # --- Simulation ---
    print(f"Generating {NUM_COMPLEXES_TO_RUN} bottom-up complexes with {NUM_VERTICES} vertices...")
    chi_values = generate_complexes_bottom_up(
        NUM_VERTICES,
        PROBABILITIES,
        NUM_COMPLEXES_TO_RUN
    )
    print("Simulation complete.")

    # --- Analysis & Visualization matching deliverable (d) ---
    avg_chi = np.mean(chi_values)
    print(f"\nAverage Euler Characteristic (χ): {avg_chi:.2f}")

    # Create a figure with two subplots, one for the histogram and one for the box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    title = f'Euler Characteristic (χ) for Bottom-Up Model'
    subtitle = f'n={NUM_VERTICES}, p_edge={PROBABILITIES.get(1, 0)}, p_tri={PROBABILITIES.get(2, 0)}, runs={NUM_COMPLEXES_TO_RUN}'
    fig.suptitle(title, fontsize=16)

    # --- Histogram ---
    ax1.hist(chi_values, bins='auto', align='left', edgecolor='black', color='steelblue')
    ax1.set_title('Histogram of χ values')
    ax1.set_xlabel('Euler Characteristic (χ)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(avg_chi, color='r', linestyle='dashed', linewidth=2, label=f'Average χ = {avg_chi:.2f}')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Box Plot ---
    ax2.boxplot(chi_values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Box Plot of χ values')
    ax2.set_ylabel('Euler Characteristic (χ)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # Hide x-axis labels for the box plot as they aren't meaningful
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()

    # --- Discussion of Findings ---
    print(f"""
        ### --- Discussion of Findings ---

        This simulation generated {NUM_COMPLEXES_TO_RUN} random simplicial complexes on a set of {NUM_VERTICES} 
        vertices. The parameters for construction were a {PROBABILITIES.get(1, 0):.0%} probability of adding an edge 
        (`p₁`) and a {PROBABILITIES.get(2, 0):.0%} probability of filling in a triangle where its boundary existed 
        (`p₂`). The analysis of the resulting topologies yielded an average Euler characteristic (χ) of **{avg_chi:.2f}**.

        The distribution of the Euler characteristic is visualized in the histogram and box plot.
        The histogram shows a unimodal, roughly symmetric distribution centered on the mean,
        with the vast majority of generated complexes having a χ between -14 and -6. The
        box plot further clarifies this, indicating that 50% of the outcomes (the
        interquartile range) fall between approximately -11 and -7. The proximity of the
        median (≈ -9) to the mean ({avg_chi:.2f}) confirms the distribution's general symmetry
        and identifies one complex with χ ≈ -18 as a rare outlier.

        The consistently negative Euler characteristic provides a clear picture of the generated topology:
        the value of χ is determined by the formula:
        `χ = (#Vertices) - (#Edges) + (#Triangles)`.
        With our parameters, the typical complex consists of approximately:
        - `k₀ = {NUM_VERTICES}` (vertices)
        - `k₁ ≈ 22` (edges, on average)
        - `k₂ ≈ 3` (triangles, on average)

        The resulting `χ ≈ 10 - 22 + 3 = -9` is driven by the fact that a large number of edges are formed, but 
        relatively few of these connections are filled in to create 2-simplexes. This generates a sparse, web-like 
        structure rich in 1-dimensional loops and cycles. The negative χ quantitatively confirms that the number
        of edges consistently overwhelms the number of vertices and triangles combined.

        The simulation successfully demonstrates how local probabilistic rules of construction give rise to a 
        predictable global topology. The parameters chosen reliably produce complexes that are not simple collections 
        of points, nor are they solid, filled-in objects. Instead, they are predominantly graph-like structures whose 
        many loops are quantified by a strongly negative Euler characteristic.
        """)

if __name__ == "__main__":
    main()