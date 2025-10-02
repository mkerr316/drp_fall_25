# In geometry.py

from typing import Tuple, List, Set, TypeVar

from .fields import Field
from .interfaces import AbstractSimplex
from .linear_algebra import Vector, VectorSpace

FieldElement = TypeVar('FieldElement')


# === FIELD-AGNOSTIC LINEAR ALGEBRA HELPERS ===
# These functions replace the numpy.linalg calls to ensure all math
# respects the field's arithmetic (e.g., modulo for PrimeField).

def _gauss_elim_rank(matrix: List[List[FieldElement]], field: Field) -> int:
    """Calculates the rank of a matrix using Gaussian elimination over a given field."""
    num_rows = len(matrix)
    if num_rows == 0:
        return 0
    num_cols = len(matrix[0])
    rank = 0
    pivot_row = 0
    for j in range(num_cols):  # Iterate through columns
        if pivot_row >= num_rows:
            break
        i = pivot_row
        while i < num_rows and matrix[i][j] == field.zero:
            i += 1

        if i < num_rows:
            matrix[pivot_row], matrix[i] = matrix[i], matrix[pivot_row]  # Swap rows
            pivot_val = matrix[pivot_row][j]
            inv_pivot = field.inverse(pivot_val)
            # Normalize pivot row
            for k in range(j, num_cols):
                matrix[pivot_row][k] = field.mul(matrix[pivot_row][k], inv_pivot)

            # Eliminate other entries in this column
            for i in range(num_rows):
                if i != pivot_row:
                    factor = matrix[i][j]
                    for k in range(j, num_cols):
                        term = field.mul(factor, matrix[pivot_row][k])
                        matrix[i][k] = field.sub(matrix[i][k], term)
            pivot_row += 1
            rank += 1
    return rank


class Simplex(AbstractSimplex):
    """
    Represents a k-simplex, defined as the convex hull of k+1 affinely independent
    vertices. This implementation is field-agnostic and suitable for TDA.
    """

    def __init__(self, vertices: Set[Vector]):
        if not vertices:
            raise ValueError("A simplex requires at least one vertex.")

        self._vertices = tuple(vertices)
        self._space = self._vertices[0].space
        self._dimension = len(vertices) - 1

        # --- VALIDATION: Check for affine independence over the given field ---
        if self.dimension > 0:
            v0 = self._vertices[0]
            field = self.space.field

            # Create a list of lists from vector contents
            edge_vectors_contents = [
                list((v - v0).contents) for v in self._vertices[1:]
            ]

            # Note: We check the rank of the row vectors (or column vectors, it's the same).
            # The matrix for rank check should have k rows (vectors) and n columns (dimension).
            matrix_for_rank = [row[:] for row in edge_vectors_contents]  # Make a copy for rank check
            rank = _gauss_elim_rank(matrix_for_rank, field)

            if rank != self.dimension:
                raise ValueError(
                    f"The {len(vertices)} provided vertices are not affinely independent "
                    f"over the field {field}. Expected rank {self.dimension}, got {rank}."
                )

    @property
    def space(self) -> VectorSpace:
        return self._space

    @property
    def field(self) -> Field:
        return self.space.field

    @property
    def vertices(self) -> Tuple[Vector, ...]:
        return self._vertices

    @property
    def dimension(self) -> int:
        return self._dimension

    def __contains__(self, point: Vector) -> bool:
        """
        Checks if a point is inside the simplex. For real spaces, this uses
        barycentric coordinates. For finite fields, it checks if the point
        lies on the affine hull spanned by the vertices.
        """
        if not isinstance(point, Vector) or point.space != self.space:
            return False
        if self.dimension < 0:
            return False  # Empty simplex
        if self.dimension == 0:
            return point.contents == self.vertices[0].contents

        v0 = self.vertices[0]
        field = self.space.field

        # System to solve: A * c = b
        # where A's columns are the edge vectors (v_i - v_0)
        # and b is the vector (point - v_0)
        A_T = [list((self.vertices[i] - v0).contents) for i in range(1, self.dimension + 1)]
        A = [list(i) for i in zip(*A_T)]  # Transpose A
        b = list((point - v0).contents)

        # For a finite field, any point in the affine hull is considered "contained".
        # We just need to check if the system Ax=b has a solution.
        # We can do this by comparing the rank of A and the rank of the augmented matrix [A|b].

        # Create augmented matrix [A|b]
        augmented_matrix = [A[i] + [b[i]] for i in range(len(A))]

        # Create copies for rank calculation, as our function modifies the matrix
        matrix_A_copy = [row[:] for row in A]
        matrix_aug_copy = [row[:] for row in augmented_matrix]

        rank_A = _gauss_elim_rank(matrix_A_copy, field)
        rank_aug = _gauss_elim_rank(matrix_aug_copy, field)

        # By the Rouché-Capelli theorem, a solution exists if and only if rank(A) == rank([A|b])
        return rank_A == rank_aug

    def __repr__(self) -> str:
        return f"{self.dimension}-Simplex with vertices at {self.vertices}"