"""Stage 2: Frozen RNS Circle Encoding.

Maps integer digit vectors to the Residue Number System circle representation.
Each integer is encoded as a tuple of (cos, sin) pairs — one per prime modulus —
representing points on unit circles. All weights are frozen (precomputed constants).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from mathllm.arb.constants import compute_coefficient_matrix, compute_exp_coefficient_matrix


class RNSCircleEncoder(nn.Module):
    """Encode digit vectors into RNS circle representation (frozen).

    Given a digit vector d = (d_0, d_1, ..., d_{K-1}) representing
    n = sum(d_k * 10^k), computes for each prime p_i:
        r_i = sum(d_k * (10^k mod p_i))  (via frozen matrix multiply)
        circle_i = (cos(2*pi*r_i/p_i), sin(2*pi*r_i/p_i))

    The trig functions absorb modular reduction: cos/sin are periodic with period p_i
    when the argument is 2*pi*r/p_i, so explicit mod is never needed.
    """

    def __init__(self, primes: tuple[int, ...], num_digits: int = 10):
        super().__init__()
        self.primes = primes
        self.num_primes = len(primes)
        self.num_digits = num_digits

        # Frozen coefficient matrix: C[i, k] = (10^k) mod p_i
        self.register_buffer(
            "coeff_matrix", compute_coefficient_matrix(primes, num_digits)
        )
        # Frozen exponent coefficient matrix: C_exp[i, k] = (10^k) mod (p_i - 1)
        self.register_buffer(
            "exp_coeff_matrix", compute_exp_coefficient_matrix(primes, num_digits)
        )
        # Precompute 2*pi/p_i for each prime
        self.register_buffer(
            "two_pi_over_p",
            torch.tensor(
                [2.0 * math.pi / p for p in primes], dtype=torch.float32
            ),
        )
        # Precompute 2*pi/(p_i - 1) for exponent encoding
        self.register_buffer(
            "two_pi_over_pm1",
            torch.tensor(
                [2.0 * math.pi / (p - 1) for p in primes], dtype=torch.float32
            ),
        )

    def forward(self, digits: Tensor) -> Tensor:
        """Encode digit vector to RNS circle representation.

        Args:
            digits: [batch, seq_len, K] digit vectors (integers in [0, 9])

        Returns:
            circle: [batch, seq_len, m, 2] where m = num_primes,
                     last dim is (cos, sin) on unit circle per prime
        """
        # Weighted residues: r[..., i] = C[i, :] @ digits[..., :]
        # digits: [B, S, K], coeff_matrix: [m, K] -> residues: [B, S, m]
        residues = torch.matmul(digits, self.coeff_matrix.T)

        # Angles: theta_i = 2*pi * r_i / p_i
        # two_pi_over_p: [m] -> broadcast with residues: [B, S, m]
        angles = residues * self.two_pi_over_p

        # Circle encoding
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        return torch.stack([cos_vals, sin_vals], dim=-1)  # [B, S, m, 2]

    def encode_exponent(self, digits: Tensor) -> Tensor:
        """Encode exponent digit vector, reducing mod (p_i - 1) via Fermat's Little Theorem.

        For exponentiation a^b mod p_i, the exponent b is reduced mod (p_i - 1)
        because a^(p-1) ≡ 1 (mod p) for prime p (Fermat's Little Theorem).

        Args:
            digits: [batch, seq_len, K] exponent digit vectors

        Returns:
            circle: [batch, seq_len, m, 2] circle encoding with Fermat-reduced exponents
        """
        residues = torch.matmul(digits, self.exp_coeff_matrix.T)
        angles = residues * self.two_pi_over_pm1
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        return torch.stack([cos_vals, sin_vals], dim=-1)
