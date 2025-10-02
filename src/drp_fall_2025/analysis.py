import copy
from collections import defaultdict

from src.drp_fall_2025.fields import PrimeField
from src.drp_fall_2025.geometry import _gauss_elim_rank
from src.drp_fall_2025.topology import SimplicialComplex


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

def _get_simplices_by_dim(simplicial_complex: SimplicialComplex) -> dict[int, list[frozenset[int]]]:
    """Groups simplices by their dimension and sorts them."""
    simplices_by_dim = defaultdict(list)
    for simplex in simplicial_complex.simplices:
        dim = len(simplex) - 1
        if dim >= 0:
            simplices_by_dim[dim].append(simplex)

    # Sort for consistent matrix column/row ordering
    for dim in simplices_by_dim:
        simplices_by_dim[dim].sort(key=lambda s: tuple(sorted(list(s))))
    return simplices_by_dim

def _get_boundary_matrix(
        k: int,
        simplices_by_dim: dict[int, list[frozenset[int]]],
        field: 'PrimeField'
) -> list[list[int]]:
    """
    Constructs the boundary matrix for dimension k over F₂.

    Rows correspond to (k-1)-simplices, columns to k-simplices.
    """
    k_simplices = simplices_by_dim.get(k, [])
    k_minus_1_simplices = simplices_by_dim.get(k - 1, [])

    if not k_simplices or not k_minus_1_simplices:
        return []

    # Create a mapping from simplex to its index for quick lookups
    k_minus_1_map = {simplex: i for i, simplex in enumerate(k_minus_1_simplices)}

    num_rows = len(k_minus_1_simplices)
    num_cols = len(k_simplices)

    matrix = [[field.zero for _ in range(num_cols)] for _ in range(num_rows)]

    for j, k_simplex in enumerate(k_simplices):
        # The boundary of a k-simplex is the sum of its (k-1)-faces
        for i in range(len(k_simplex)):
            face = frozenset(list(k_simplex)[:i] + list(k_simplex)[i + 1:])
            if face in k_minus_1_map:
                row_idx = k_minus_1_map[face]
                # In F₂, adding 1 is the same as flipping the bit.
                # Since we are working mod 2, we just set it to 1.
                matrix[row_idx][j] = field.one

    return matrix

def compute_betti_numbers(simplicial_complex: SimplicialComplex) -> dict[int, int]:
    """
    Computes the Betti numbers for a given simplicial complex over F₂.

    :param simplicial_complex: An instance of the SimplicialComplex class.
    :return: A dictionary mapping dimension k to the k-th Betti number βₖ.
    """
    field = PrimeField(2)
    simplices_by_dim = _get_simplices_by_dim(simplicial_complex)

    if not simplices_by_dim:
        return {0: 0}

    max_dim = max(simplices_by_dim.keys())

    # 1. Compute all boundary matrices
    boundary_matrices = {}
    for k in range(1, max_dim + 2):  # Go one dim higher for the final image
        boundary_matrices[k] = _get_boundary_matrix(k, simplices_by_dim, field)

    # 2. Compute ranks of all boundary matrices
    ranks = {}
    for k, matrix in boundary_matrices.items():
        if not matrix:
            ranks[k] = 0
        else:
            # IMPORTANT: _gauss_elim_rank modifies the matrix in place.
            # We must pass a copy to preserve the original matrix.
            matrix_copy = copy.deepcopy(matrix)
            ranks[k] = _gauss_elim_rank(matrix_copy, field)

    # 3. Compute Betti numbers using the rank-nullity theorem
    betti_numbers = {}

    # Betti 0: Number of connected components
    # β₀ = dim(ker ∂₀) - dim(im ∂₁)
    # dim(ker ∂₀) is the number of 0-simplices (vertices)
    num_0_simplices = len(simplices_by_dim.get(0, []))
    dim_im_d1 = ranks.get(1, 0)
    betti_numbers[0] = num_0_simplices - dim_im_d1

    # Betti k for k > 0
    for k in range(1, max_dim + 1):
        num_k_simplices = len(simplices_by_dim.get(k, []))
        dim_ker_dk = num_k_simplices - ranks.get(k, 0)
        dim_im_dk_plus_1 = ranks.get(k + 1, 0)

        betti_numbers[k] = dim_ker_dk - dim_im_dk_plus_1

    return betti_numbers