from abc import ABC, abstractmethod
import math

# _is_prime function remains the same...
def _is_prime(n: int) -> bool:
    if n <= 1: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

class Field(ABC):
    @abstractmethod
    def add(self, a, b): # Renamed from __add__
        pass

    @abstractmethod
    def mul(self, a, b): # Renamed from __mul__
        pass

    @abstractmethod
    def neg(self, a): # Renamed from __neg__
        pass

    @abstractmethod
    def inverse(self, a):
        """Multiplicative inverse"""
        pass

    def sub(self, a, b): # Renamed from __sub__
        return self.add(a, self.neg(b))

    def truediv(self, a, b): # Renamed from __truediv__
        return self.mul(a, self.inverse(b))

    @property
    @abstractmethod
    def zero(self):
        pass

    @property
    @abstractmethod
    def one(self):
        pass

class PrimeField(Field):
    """Prime finite field of order p."""
    def __init__(self, order: int):
        if not isinstance(order, int):
            raise TypeError("Order must be an integer.")
        if order < 2:
            raise ValueError("Order must be greater than 1.")
        if not _is_prime(order):
            raise ValueError("Order must be prime.")
        self.order = order

    def __repr__(self) -> str:
        return f"PrimeField(order={self.order})"

    def add(self, a: int, b: int) -> int: # Renamed from __add__
        return (a + b) % self.order

    def mul(self, a: int, b: int) -> int: # Renamed from __mul__
        return (a * b) % self.order

    def neg(self, a: int) -> int: # Renamed from __neg__
        return -a % self.order

    def inverse(self, a: int) -> int:
        if a % self.order == 0:
            raise ZeroDivisionError("Division by zero in a prime field")
        return pow(a, -1, self.order)

    @property
    def zero(self):
        return 0

    @property
    def one(self):
        return 1