# In topology.py

import itertools
import random
from typing import Set, Iterable

from .interfaces import AbstractSimplicialComplex


class SimplicialComplex(AbstractSimplicialComplex):
    """
    Represents an abstract simplicial complex, defined as a downward-closed
    collection of simplices (sets of vertices).

    This class is purely combinatorial; vertices are represented by integers.
    It is independent of any geometric embedding or field arithmetic.
    """

    def __init__(self, simplices: Set[frozenset[int]]):
        """
        Initializes the simplicial complex from a set of frozensets.

        :param simplices: A set of frozensets, where each frozenset represents
                          a simplex (face) in the complex. This set must be
                          downward-closed.
        :raises ValueError: If the provided set of simplices is not
                            downward-closed.
        """
        self._simplices = simplices
        if not self._is_valid():
            raise ValueError(
                "The provided set of simplices is not downward-closed. "
                "If a simplex is in the complex, all its faces must also be present."
            )

    def _is_valid(self) -> bool:
        """Checks if the complex is downward-closed."""
        for simplex in self._simplices:
            if len(simplex) > 1:
                # Check that all (k-1)-faces of this k-simplex exist
                for sub_face in itertools.combinations(simplex, len(simplex) - 1):
                    if frozenset(sub_face) not in self._simplices:
                        return False
        return True

    @classmethod
    def from_maximal_simplices(cls, maximal_simplices: Iterable[Set[int]]) -> 'SimplicialComplex':
        """
        Creates a valid simplicial complex by generating all sub-faces from a
        list of its maximal faces (facets).

        :param maximal_simplices: An iterable of sets, where each set contains
                                  the integer vertices of a maximal simplex.
        """
        all_simplices = set()
        for maximal_simplex in maximal_simplices:
            if not maximal_simplex: continue
            for k in range(1, len(maximal_simplex) + 1):
                for face in itertools.combinations(maximal_simplex, k):
                    all_simplices.add(frozenset(face))
        # Add the empty set if you consider it a (-1)-simplex, otherwise omit.
        # For most TDA purposes, we start with 0-simplices.
        # Add vertices (0-simplices) if they weren't part of a higher-dim simplex
        all_vertices = set().union(*maximal_simplices)
        for v in all_vertices:
            all_simplices.add(frozenset([v]))

        return cls(all_simplices)

    # Your other factory methods are well-designed and can remain as they are.
    # ... from_top_down_process and from_bottom_up_process ...
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

        return cls.from_maximal_simplices([set(s) for s in current_maximals])

    @classmethod
    def from_bottom_up_process(cls, num_vertices: int, p_dict: dict[int, float]) -> 'SimplicialComplex':
        """
        Creates a random complex using a bottom-up process (Gilbert-style model).

        Simplices are added probabilistically at each dimension, provided their
        boundaries already exist in the complex.

        :param num_vertices: The total number of vertices (labeled 0 to n-1).
        :param p_dict: A dictionary mapping dimension 'k' to the probability
                       p_k of including a k-simplex.
        """
        simplices = {frozenset([i]) for i in range(num_vertices)}

        for k in range(1, num_vertices):
            pk = p_dict.get(k, 0.0)
            if pk == 0: continue

            for candidate_tuple in itertools.combinations(range(num_vertices), k + 1):
                candidate = frozenset(candidate_tuple)

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
        if not self.simplices:
            return -1  # Dimension of an empty complex
        # Filter out the empty set if it were ever included
        non_empty_simplices = {s for s in self.simplices if s}
        if not non_empty_simplices:
            return -1
        return max(len(s) for s in non_empty_simplices) - 1

    def __repr__(self) -> str:
        num_simplices = len(self.simplices)
        if not self.simplices:
            return "SimplicialComplex(vertices=0, simplices=0, dim=-1)"

        all_vertices = set().union(*self.simplices) if self.simplices else set()
        num_vertices = len(all_vertices)
        return f"SimplicialComplex(vertices={num_vertices}, simplices={num_simplices}, dim={self.dimension})"