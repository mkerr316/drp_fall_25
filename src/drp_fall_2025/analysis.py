from collections import defaultdict

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
