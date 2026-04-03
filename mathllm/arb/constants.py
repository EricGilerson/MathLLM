"""Precomputed mathematical constants for the Arithmetic Residual Block.

All frozen tensors used in Stages 2 and 3 are constructed here:
- RNS coefficient matrices
- CRT reconstruction weights
- Multiplication and exponentiation lookup tables
- Circle template vectors for residue decoding
"""

from __future__ import annotations

import math
from functools import reduce

import torch
from torch import Tensor

# Default prime moduli for RNS. Product ≈ 7.42 × 10^9.
DEFAULT_PRIMES = (7, 11, 13, 17, 19, 23, 29, 31, 37)
DEFAULT_NUM_DIGITS = 10  # K: digit slots, covers integers up to ~10^10


def mod_inverse(a: int, p: int) -> int:
    """Compute modular multiplicative inverse of a mod p using extended Euclidean algorithm.

    Requires gcd(a, p) == 1 (guaranteed when p is prime and a % p != 0).
    """
    g, x, _ = _extended_gcd(a % p, p)
    if g != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {p}) = {g}")
    return x % p


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def compute_product(primes: tuple[int, ...]) -> int:
    """Compute the product P of all primes (the RNS representable range)."""
    return reduce(lambda a, b: a * b, primes)


def integer_to_digits(n: int, K: int) -> list[int]:
    """Decompose non-negative integer n into base-10 digit vector (least-significant first)."""
    digits = []
    val = abs(n)
    for _ in range(K):
        digits.append(val % 10)
        val //= 10
    return digits


def digits_to_integer(digits: list[int] | Tensor) -> int:
    """Reconstruct integer from base-10 digit vector (least-significant first)."""
    if isinstance(digits, Tensor):
        digits = digits.long().tolist()
    result = 0
    for k in range(len(digits) - 1, -1, -1):
        result = result * 10 + digits[k]
    return result


def compute_coefficient_matrix(
    primes: tuple[int, ...], K: int = DEFAULT_NUM_DIGITS
) -> Tensor:
    """Compute the frozen RNS coefficient matrix C where C[i, k] = (10^k) mod p_i.

    Used in Stage 2 to map digit vectors to weighted residues:
        r_i = sum_k(d_k * 10^k mod p_i) = C[i, :] @ d

    Returns: Tensor of shape [m, K] (float32)
    """
    m = len(primes)
    C = torch.zeros(m, K, dtype=torch.float32)
    for i, p in enumerate(primes):
        for k in range(K):
            C[i, k] = pow(10, k, p)
    return C


def compute_exp_coefficient_matrix(
    primes: tuple[int, ...], K: int = DEFAULT_NUM_DIGITS
) -> Tensor:
    """Compute the exponent reduction coefficient matrix C_exp where C_exp[i, k] = (10^k) mod (p_i - 1).

    Used for Fermat's Little Theorem: a^b mod p = a^(b mod (p-1)) mod p.
    The exponent b is reduced mod (p_i - 1) via: b mod (p_i-1) = C_exp[i, :] @ digits_of_b

    Returns: Tensor of shape [m, K] (float32)
    """
    m = len(primes)
    C_exp = torch.zeros(m, K, dtype=torch.float32)
    for i, p in enumerate(primes):
        mod = p - 1  # Fermat modulus
        for k in range(K):
            C_exp[i, k] = pow(10, k, mod)
    return C_exp


def compute_crt_weights(primes: tuple[int, ...]) -> Tensor:
    """Compute CRT reconstruction weights.

    For each prime p_i, compute w_i = M_i * (M_i^{-1} mod p_i)
    where M_i = P / p_i and P = product of all primes.

    The integer n is reconstructed as: n = (sum_i r_i * w_i) mod P

    Returns: Tensor of shape [m] (float64 to avoid overflow — weights can be ~7.4e9)
    """
    P = compute_product(primes)
    weights = []
    for p in primes:
        M_i = P // p
        M_i_inv = mod_inverse(M_i, p)
        weights.append(M_i * M_i_inv)
    return torch.tensor(weights, dtype=torch.float64)


def compute_multiplication_tables(primes: tuple[int, ...]) -> list[Tensor]:
    """Compute frozen multiplication lookup tables for each prime.

    T_p[a, b] = (a * b) mod p for a, b in {0, ..., p-1}.

    Returns: list of m tensors, each of shape [p_i, p_i] (float32)
    """
    tables = []
    for p in primes:
        T = torch.zeros(p, p, dtype=torch.float32)
        for a in range(p):
            for b in range(p):
                T[a, b] = (a * b) % p
        tables.append(T)
    return tables


def compute_exponentiation_tables(primes: tuple[int, ...]) -> list[Tensor]:
    """Compute frozen exponentiation lookup tables for each prime.

    T_p[a, k] = a^k mod p for a in {0, ..., p-1}, k in {0, ..., p-2}.
    (k ranges over 0..p-2 because by Fermat, a^(p-1) ≡ 1, so exponents reduce mod p-1.)

    Special handling for k=0: we store a^(p-1) mod p instead of a^0.
    This is because Fermat reduction maps both b=0 and b=p-1 to index 0,
    and for a ≡ 0 mod p, a^(p-1) = 0 while a^0 = 1. We handle b=0 (a^0 = 1)
    as a special case in circle_exp.

    Returns: list of m tensors, each of shape [p_i, p_i - 1] (float32)
    """
    tables = []
    for p in primes:
        exp_range = p - 1  # exponents 0 .. p-2
        T = torch.zeros(p, exp_range, dtype=torch.float32)
        for a in range(p):
            # k=0: store a^(p-1) mod p (not a^0) to correctly handle
            # b > 0 where b ≡ 0 mod (p-1)
            T[a, 0] = pow(a, p - 1, p)
            for k in range(1, exp_range):
                T[a, k] = pow(a, k, p)
        tables.append(T)
    return tables


def compute_circle_templates(primes: tuple[int, ...]) -> list[Tensor]:
    """Compute unit circle template vectors for residue decoding.

    For each prime p, produces p template vectors:
        template[r] = (cos(2*pi*r/p), sin(2*pi*r/p))  for r = 0, ..., p-1

    Inner product of a circle-encoded residue with these templates yields
    maximum similarity at the correct residue index.

    Returns: list of m tensors, each of shape [p_i, 2] (float32)
    """
    templates = []
    for p in primes:
        angles = torch.tensor(
            [2.0 * math.pi * r / p for r in range(p)], dtype=torch.float32
        )
        T = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        templates.append(T)
    return templates


def compute_digit_decomposition_weights(K: int = DEFAULT_NUM_DIGITS) -> Tensor:
    """Compute powers of 10 for digit decomposition: [1, 10, 100, ..., 10^(K-1)].

    Used to convert digit vectors back to integers: n = powers @ digits.

    Returns: Tensor of shape [K] (float64)
    """
    return torch.tensor([10**k for k in range(K)], dtype=torch.float64)
