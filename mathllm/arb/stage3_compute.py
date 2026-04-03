"""Stage 3: Frozen Arithmetic Computation.

Executes addition, subtraction, multiplication, exponentiation, and division
in parallel using the RNS circle encoding. All weights are frozen.

- Addition: complex multiplication of circle encodings (exact modular addition)
- Subtraction: conjugate + complex multiplication (exact modular subtraction)
- Multiplication: decode residues -> lookup table -> re-encode
- Exponentiation: Fermat reduction + lookup table
- Division: decode residues -> modular inverse lookup table -> re-encode
- CRT reconstruction: dot product with frozen weights to recover integer
- Digit decomposition: integer -> base-10 digit vector
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mathllm.arb.constants import (
    compute_circle_templates,
    compute_crt_weights,
    compute_division_tables,
    compute_exponentiation_tables,
    compute_multiplication_tables,
    compute_product,
)


class ArithmeticCompute(nn.Module):
    """Frozen computation of +, -, *, exp, / in RNS circle encoding."""

    def __init__(
        self,
        primes: tuple[int, ...],
        num_digits: int = 10,
        softmax_temperature: float = 1000.0,
        repair_division_during_training: bool = True,
    ):
        super().__init__()
        self.primes = primes
        self.num_primes = len(primes)
        self.num_digits = num_digits
        self.softmax_temperature = softmax_temperature
        self.repair_division_during_training = repair_division_during_training
        self.P = compute_product(primes)

        # Keep the registered buffer MPS-safe. Exact CRT reconstruction uses a
        # lazily materialized float64 copy off the module state.
        crt_weights = compute_crt_weights(primes)
        self.register_buffer("crt_weights", crt_weights.to(dtype=torch.float32))
        self._crt_weights_fp64_cpu = crt_weights
        self._crt_weights_fp64_cache: dict[str, Tensor] = {}

        # Circle templates for residue decoding
        templates = compute_circle_templates(primes)
        for i, t in enumerate(templates):
            self.register_buffer(f"templates_{i}", t)

        # Multiplication tables — stored as one-hot indexed tensors
        # For each prime p, mul_onehot[a, b, :] is a one-hot vector of length p
        # with 1 at position (a*b) mod p. This allows soft lookup via einsum.
        mul_tables = compute_multiplication_tables(primes)
        for i, (p, t) in enumerate(zip(primes, mul_tables)):
            # t: [p, p] with integer values in [0, p-1]
            # Convert to one-hot: [p, p, p]
            t_long = t.long()
            onehot = torch.zeros(p, p, p, dtype=torch.float32)
            for a in range(p):
                for b in range(p):
                    onehot[a, b, t_long[a, b]] = 1.0
            self.register_buffer(f"mul_onehot_{i}", onehot)

        # Division tables — same one-hot treatment as multiplication
        div_tables = compute_division_tables(primes)
        for i, (p, t) in enumerate(zip(primes, div_tables)):
            t_long = t.long()
            onehot = torch.zeros(p, p, p, dtype=torch.float32)
            for a in range(p):
                for b in range(p):
                    onehot[a, b, t_long[a, b]] = 1.0
            self.register_buffer(f"div_onehot_{i}", onehot)

        # Exponentiation tables — same one-hot treatment
        exp_tables = compute_exponentiation_tables(primes)
        for i, (p, t) in enumerate(zip(primes, exp_tables)):
            pm1 = p - 1
            t_long = t.long()
            onehot = torch.zeros(p, pm1, p, dtype=torch.float32)
            for a in range(p):
                for k in range(pm1):
                    onehot[a, k, t_long[a, k]] = 1.0
            self.register_buffer(f"exp_onehot_{i}", onehot)

        # Precompute 2*pi/p for re-encoding results to circles
        self.register_buffer(
            "two_pi_over_p",
            torch.tensor([2.0 * math.pi / p for p in primes], dtype=torch.float32),
        )

        # Precompute exponent templates for circle_exp (avoids per-forward allocation)
        # For each prime p_i, store (p_i - 1) evenly-spaced unit-circle points
        for i, p in enumerate(primes):
            pm1 = p - 1
            angles = torch.arange(pm1, dtype=torch.float32) * (2.0 * math.pi / pm1)
            exp_tmpl = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [pm1, 2]
            self.register_buffer(f"exp_templates_{i}", exp_tmpl)

    def _get_template(self, i: int) -> Tensor:
        return getattr(self, f"templates_{i}")

    def _get_mul_onehot(self, i: int) -> Tensor:
        return getattr(self, f"mul_onehot_{i}")

    def _get_div_onehot(self, i: int) -> Tensor:
        return getattr(self, f"div_onehot_{i}")

    def _get_exp_onehot(self, i: int) -> Tensor:
        return getattr(self, f"exp_onehot_{i}")

    def _get_exp_template(self, i: int) -> Tensor:
        return getattr(self, f"exp_templates_{i}")

    def _get_crt_weights_fp64(self, device: torch.device) -> Tensor:
        """Return float64 CRT weights on a backend that supports them."""
        if device.type in {"cpu", "mps"}:
            return self._crt_weights_fp64_cpu

        cache_key = str(device)
        weights = self._crt_weights_fp64_cache.get(cache_key)
        if weights is None:
            weights = self._crt_weights_fp64_cpu.to(device=device)
            self._crt_weights_fp64_cache[cache_key] = weights
        return weights

    # ------------------------------------------------------------------
    # Addition: complex multiplication of circle encodings
    # ------------------------------------------------------------------

    def circle_add(self, a_circle: Tensor, b_circle: Tensor) -> Tensor:
        """Modular addition via complex multiplication.

        (cos a + i sin a)(cos b + i sin b) = cos(a+b) + i sin(a+b)

        Args:
            a_circle, b_circle: [batch, seq, m, 2] — (cos, sin) per prime

        Returns:
            [batch, seq, m, 2] — circle encoding of (a + b) mod p_i
        """
        ca, sa = a_circle[..., 0], a_circle[..., 1]
        cb, sb = b_circle[..., 0], b_circle[..., 1]
        cos_sum = ca * cb - sa * sb
        sin_sum = sa * cb + ca * sb
        return torch.stack([cos_sum, sin_sum], dim=-1)

    # ------------------------------------------------------------------
    # Subtraction: conjugate second operand, then complex multiply
    # ------------------------------------------------------------------

    def circle_sub(self, a_circle: Tensor, b_circle: Tensor) -> Tensor:
        """Modular subtraction via conjugation + complex multiplication.

        a - b = a + (-b), and -b on the circle is the conjugate (negate sin).

        Args:
            a_circle, b_circle: [batch, seq, m, 2]

        Returns:
            [batch, seq, m, 2] — circle encoding of (a - b) mod p_i
        """
        b_conj = torch.stack([b_circle[..., 0], -b_circle[..., 1]], dim=-1)
        return self.circle_add(a_circle, b_conj)

    # ------------------------------------------------------------------
    # Residue decoding: circle -> integer residue per prime
    # ------------------------------------------------------------------

    def _decode_residues_soft(self, circle: Tensor) -> list[Tensor]:
        """Decode circle encodings to soft one-hot residue distributions.

        For each prime p_i, computes inner product of the circle encoding with
        p_i template vectors, then applies softmax with high temperature to get
        a differentiable approximation to one-hot residue selection.

        Args:
            circle: [batch, seq, m, 2]

        Returns:
            List of m tensors, each [batch, seq, p_i] — soft one-hot over residues
        """
        soft_residues = []
        for i, p in enumerate(self.primes):
            c_i = circle[:, :, i, :]  # [B, S, 2]
            templates = self._get_template(i)  # [p, 2]
            # Inner product: [B, S, 2] @ [2, p] -> [B, S, p]
            similarity = torch.matmul(c_i, templates.T)
            soft = F.softmax(similarity * self.softmax_temperature, dim=-1)
            soft_residues.append(soft)
        return soft_residues

    def _decode_residues_hard(self, circle: Tensor) -> Tensor:
        """Decode circle encodings to integer residues (non-differentiable).

        Args:
            circle: [batch, seq, m, 2]

        Returns:
            [batch, seq, m] — integer residues per prime
        """
        residues = []
        for i, p in enumerate(self.primes):
            c_i = circle[:, :, i, :]  # [B, S, 2]
            templates = self._get_template(i)  # [p, 2]
            similarity = torch.matmul(c_i, templates.T)  # [B, S, p]
            residues.append(similarity.argmax(dim=-1))  # [B, S]
        return torch.stack(residues, dim=-1).float()  # [B, S, m]

    # ------------------------------------------------------------------
    # Multiplication: decode -> table lookup -> re-encode
    # ------------------------------------------------------------------

    def circle_mul(self, a_circle: Tensor, b_circle: Tensor) -> Tensor:
        """Modular multiplication via residue decoding and one-hot table lookup.

        For each prime p_i:
        1. Decode both operands to soft one-hot residue distributions
        2. Outer product selects cells in the one-hot multiplication table
        3. Sum gives a soft distribution over result residues
        4. Weighted combination of circle-encoded residues gives result

        Args:
            a_circle, b_circle: [batch, seq, m, 2]

        Returns:
            [batch, seq, m, 2] — circle encoding of (a * b) mod p_i
        """
        a_soft = self._decode_residues_soft(a_circle)
        b_soft = self._decode_residues_soft(b_circle)

        result_cos = []
        result_sin = []

        for i, p in enumerate(self.primes):
            # Outer product of soft one-hots: [B, S, p_a, p_b]
            outer = torch.einsum("bsi,bsj->bsij", a_soft[i], b_soft[i])
            # One-hot table: [p, p, p] — last dim is one-hot result residue
            mul_onehot = self._get_mul_onehot(i)  # [p, p, p]
            # Contract outer product with table to get result distribution
            # outer: [B, S, p, p], table: [p, p, p] -> result_dist: [B, S, p]
            result_dist = torch.einsum("bsij,ijk->bsk", outer, mul_onehot)

            # Weighted sum of circle templates to get result circle encoding
            templates = self._get_template(i)  # [p, 2]
            # result_dist: [B, S, p], templates: [p, 2] -> [B, S, 2]
            result_circle_i = torch.matmul(result_dist, templates)

            result_cos.append(result_circle_i[..., 0])
            result_sin.append(result_circle_i[..., 1])

        cos_stack = torch.stack(result_cos, dim=-1)  # [B, S, m]
        sin_stack = torch.stack(result_sin, dim=-1)
        return torch.stack([cos_stack, sin_stack], dim=-1)  # [B, S, m, 2]

    # ------------------------------------------------------------------
    # Exponentiation: Fermat reduction + table lookup
    # ------------------------------------------------------------------

    def circle_exp(
        self, base_circle: Tensor, exp_circle: Tensor
    ) -> Tensor:
        """Exponentiation via Fermat's Little Theorem and table lookup.

        base_circle encodes the base a (reduced mod p_i).
        exp_circle encodes the exponent b (reduced mod (p_i - 1) by Stage 2).

        For each prime p_i:
        1. Decode base residue (mod p_i) from base_circle
        2. Decode exponent residue (mod p_i - 1) from exp_circle
        3. Lookup in exponentiation table T[a, k] = a^k mod p_i
        4. Re-encode result to circle

        Special case: b=0 means a^0 = 1 for all a. Since Fermat reduction maps
        both b=0 and b=p-1 to index 0, we detect b=0 (all exponent residues are 0)
        and return circle encoding of 1 in that case. The table stores a^(p-1) at
        index 0 so that b > 0 with b ≡ 0 mod (p-1) works correctly.

        Args:
            base_circle: [batch, seq, m, 2] — circle encoding of base
            exp_circle: [batch, seq, m, 2] — circle encoding of exponent (Fermat-reduced)

        Returns:
            [batch, seq, m, 2] — circle encoding of base^exp mod p_i
        """
        base_soft = self._decode_residues_soft(base_circle)

        # Decode exponent residues — use pre-cached (p_i - 1) templates
        exp_soft = []
        for i, p in enumerate(self.primes):
            e_i = exp_circle[:, :, i, :]  # [B, S, 2]
            exp_templates = self._get_exp_template(i)  # [pm1, 2] — pre-cached buffer
            similarity = torch.matmul(e_i, exp_templates.T)  # [B, S, pm1]
            soft = F.softmax(similarity * self.softmax_temperature, dim=-1)
            exp_soft.append(soft)

        # Detect b=0: when b=0, ALL exponent residues decode to 0.
        # Compute probability that b=0 as the product of p(residue=0) across primes.
        b_zero_prob = exp_soft[0][:, :, 0]  # [B, S]
        for i in range(1, self.num_primes):
            b_zero_prob = b_zero_prob * exp_soft[i][:, :, 0]
        # b_zero_prob: [B, S] — close to 1 when b=0, close to 0 otherwise

        result_cos = []
        result_sin = []

        for i, p in enumerate(self.primes):
            # Outer product: [B, S, p, pm1]
            outer = torch.einsum("bsi,bsj->bsij", base_soft[i], exp_soft[i])
            # One-hot exp table: [p, pm1, p]
            exp_onehot = self._get_exp_onehot(i)
            # Contract: [B, S, p, pm1] x [p, pm1, p] -> [B, S, p]
            result_dist = torch.einsum("bsij,ijk->bsk", outer, exp_onehot)

            # Weighted sum of circle templates
            templates = self._get_template(i)  # [p, 2]
            result_circle_i = torch.matmul(result_dist, templates)  # [B, S, 2]

            # For b=0: a^0 = 1, so circle encoding is template[1] = (cos(2π/p), sin(2π/p))
            one_circle = templates[1]  # [2] — circle encoding of residue 1
            # Interpolate: b_zero_prob * one_circle + (1 - b_zero_prob) * table_result
            bz = b_zero_prob.unsqueeze(-1)  # [B, S, 1]
            result_circle_i = bz * one_circle + (1.0 - bz) * result_circle_i

            result_cos.append(result_circle_i[..., 0])
            result_sin.append(result_circle_i[..., 1])

        cos_stack = torch.stack(result_cos, dim=-1)
        sin_stack = torch.stack(result_sin, dim=-1)
        return torch.stack([cos_stack, sin_stack], dim=-1)

    # ------------------------------------------------------------------
    # Division: decode -> modular inverse table lookup -> re-encode
    # ------------------------------------------------------------------

    def circle_div(self, a_circle: Tensor, b_circle: Tensor) -> Tensor:
        """Exact modular division via residue decoding and inverse table lookup.

        For each prime p_i:
        1. Decode both operands to soft one-hot residue distributions
        2. Outer product selects cells in the division table T[a,b] = a * b^{-1} mod p
        3. Sum gives a soft distribution over result residues
        4. Weighted combination of circle-encoded residues gives result

        Post-correction: when b ≡ 0 mod p_i, modular inverse doesn't exist.
        For those primes, we reconstruct the quotient from the remaining primes
        via CRT and re-encode. This handles exact divisions where b shares a
        factor with one of the RNS primes.

        Args:
            a_circle, b_circle: [batch, seq, m, 2]

        Returns:
            [batch, seq, m, 2] — circle encoding of (a / b) mod p_i
        """
        a_soft = self._decode_residues_soft(a_circle)
        b_soft = self._decode_residues_soft(b_circle)

        result_cos = []
        result_sin = []
        # Track probability that b ≡ 0 mod p_i for post-correction
        b_zero_probs = []

        for i, p in enumerate(self.primes):
            outer = torch.einsum("bsi,bsj->bsij", a_soft[i], b_soft[i])
            div_onehot = self._get_div_onehot(i)  # [p, p, p]
            result_dist = torch.einsum("bsij,ijk->bsk", outer, div_onehot)

            templates = self._get_template(i)  # [p, 2]
            result_circle_i = torch.matmul(result_dist, templates)  # [B, S, 2]

            result_cos.append(result_circle_i[..., 0])
            result_sin.append(result_circle_i[..., 1])
            b_zero_probs.append(b_soft[i][:, :, 0])  # prob b ≡ 0 mod p_i

        cos_stack = torch.stack(result_cos, dim=-1)  # [B, S, m]
        sin_stack = torch.stack(result_sin, dim=-1)
        result = torch.stack([cos_stack, sin_stack], dim=-1)  # [B, S, m, 2]
        b_zero = torch.stack(b_zero_probs, dim=-1)  # [B, S, m]

        # Post-correction: if any prime has b ≡ 0, reconstruct from good primes.
        # This is a non-differentiable correction (uses hard decode + CRT), but
        # only activates when b shares a factor with an RNS prime.
        needs_repair = (b_zero > 0.5).any(dim=-1)  # [B, S] — any prime poisoned?
        should_repair = not self.training or self.repair_division_during_training
        if should_repair and needs_repair.any():
            result = self._repair_division_residues(result, b_zero)

        return result

    def _repair_division_residues(
        self, result_circle: Tensor, b_zero_probs: Tensor,
    ) -> Tensor:
        """Repair division results for primes where b ≡ 0.

        Uses partial CRT from the good primes to reconstruct the quotient,
        then re-encodes the correct residue for the bad primes.

        Args:
            result_circle: [B, S, m, 2] — division result (some primes wrong)
            b_zero_probs: [B, S, m] — probability b ≡ 0 for each prime

        Returns:
            [B, S, m, 2] — corrected result
        """
        from mathllm.arb.constants import compute_crt_weights, mod_inverse

        B, S, m, _ = result_circle.shape
        bad_mask = b_zero_probs > 0.5  # [B, S, m]

        # Decode hard residues from the result circle
        residues = self._decode_residues_hard(result_circle)  # [B, S, m] float32

        result_out = result_circle.clone()

        for bi in range(B):
            for si in range(S):
                bad = bad_mask[bi, si]  # [m] bool
                if not bad.any():
                    continue

                good_indices = [j for j in range(m) if not bad[j]]
                if len(good_indices) < 2:
                    continue

                # Partial CRT: recompute weights for just the good primes
                good_primes = tuple(self.primes[j] for j in good_indices)
                good_crt_weights = compute_crt_weights(good_primes)  # float64
                good_residues = torch.tensor(
                    [residues[bi, si, j].item() for j in good_indices],
                    dtype=torch.float64,
                )

                good_P = 1
                for p in good_primes:
                    good_P *= p

                n = int((good_residues * good_crt_weights).sum().item() % good_P)

                # Re-encode the correct residue for each bad prime
                for j in range(m):
                    if not bad[j]:
                        continue
                    p = self.primes[j]
                    correct_residue = n % p
                    angle = 2.0 * math.pi * correct_residue / p
                    result_out[bi, si, j, 0] = math.cos(angle)
                    result_out[bi, si, j, 1] = math.sin(angle)

        return result_out

    # ------------------------------------------------------------------
    # CRT Reconstruction: residues -> integer
    # ------------------------------------------------------------------

    def crt_reconstruct(self, circle: Tensor) -> Tensor:
        """Reconstruct integer from circle encoding via Chinese Remainder Theorem.

        1. Decode circle to integer residues per prime
        2. Dot product with CRT weights (in float64 for precision)
        3. Reduce mod P

        Args:
            circle: [batch, seq, m, 2]

        Returns:
            [batch, seq] — reconstructed integers. On MPS this is returned on
            CPU because the exact float64 path is not supported by MPS.
        """
        residues = self._decode_residues_hard(circle)  # [B, S, m] float32

        # Exact CRT reconstruction needs float64. MPS cannot execute float64,
        # so perform this rare path on CPU when needed.
        if circle.device.type == "mps":
            residues_64 = residues.cpu().to(torch.float64)
            weights_64 = self._get_crt_weights_fp64(torch.device("cpu"))
        else:
            weights_64 = self._get_crt_weights_fp64(circle.device)
            residues_64 = residues.to(device=weights_64.device, dtype=torch.float64)

        weighted = residues_64 * weights_64  # [B, S, m] float64
        n = weighted.sum(dim=-1) % self.P  # [B, S]
        return n

    def crt_reconstruct_signed(self, circle: Tensor) -> Tensor:
        """CRT reconstruction with signed integer interpretation.

        Values > P/2 are interpreted as negative: n - P.

        Args:
            circle: [batch, seq, m, 2]

        Returns:
            [batch, seq] — signed integers
        """
        n = self.crt_reconstruct(circle)
        half_P = self.P / 2
        return torch.where(n > half_P, n - self.P, n)

    # ------------------------------------------------------------------
    # Digit decomposition: integer -> base-10 digit vector
    # ------------------------------------------------------------------

    def integer_to_digits(self, n: Tensor) -> Tensor:
        """Decompose integer to base-10 digit vector (least-significant first).

        Uses integer division and modulo. For gradient flow, this is wrapped
        in the full pipeline where gradients flow through the circle encoding
        path (not through this discrete step).

        Args:
            n: [batch, seq] — integers

        Returns:
            [batch, seq, K] — digit vectors (float32)
        """
        n_long = n.long().abs()
        digits = []
        remainder = n_long
        for _ in range(self.num_digits):
            digits.append((remainder % 10).float())
            remainder = remainder // 10
        return torch.stack(digits, dim=-1)  # [B, S, K]

    def integer_to_digits_with_sign(self, n: Tensor) -> Tensor:
        """Decompose signed integer to digit vector with sign indicator.

        Returns K+1 dimensions: K digits + 1 sign indicator (0 = positive, 1 = negative).

        Args:
            n: [batch, seq] — signed integers

        Returns:
            [batch, seq, K+1] — digit vectors with sign (float32)
        """
        sign = (n < 0).float().unsqueeze(-1)  # [B, S, 1]
        digits = self.integer_to_digits(n)  # [B, S, K]
        return torch.cat([digits, sign], dim=-1)  # [B, S, K+1]

    # ------------------------------------------------------------------
    # Full forward pass: compute all operations unconditionally
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def forward(
        self,
        a_circle: Tensor,
        b_circle: Tensor,
        b_exp_circle: Tensor,
    ) -> Tensor:
        """Compute all five operations unconditionally in parallel.

        Returns flattened circle encodings of all results, keeping the entire
        pipeline differentiable. The injection layer learns to interpret these
        (cos, sin) pairs.

        Args:
            a_circle: [batch, seq, m, 2] — operand A circle encoding
            b_circle: [batch, seq, m, 2] — operand B circle encoding
            b_exp_circle: [batch, seq, m, 2] — operand B exponent-reduced circle encoding

        Returns:
            results: [batch, seq, 5 * m * 2] — flattened circle encodings
                     of (add, sub, mul, exp, div) results
        """
        B, S = a_circle.shape[:2]

        # All five operations — each returns [B, S, m, 2]
        add_circle = self.circle_add(a_circle, b_circle)
        sub_circle = self.circle_sub(a_circle, b_circle)
        mul_circle = self.circle_mul(a_circle, b_circle)
        exp_circle = self.circle_exp(a_circle, b_exp_circle)
        div_circle = self.circle_div(a_circle, b_circle)

        # Flatten each: [B, S, m, 2] -> [B, S, m*2]
        add_flat = add_circle.reshape(B, S, -1)
        sub_flat = sub_circle.reshape(B, S, -1)
        mul_flat = mul_circle.reshape(B, S, -1)
        exp_flat = exp_circle.reshape(B, S, -1)
        div_flat = div_circle.reshape(B, S, -1)

        # Concatenate all: [B, S, 5 * m * 2]
        return torch.cat([add_flat, sub_flat, mul_flat, exp_flat, div_flat], dim=-1)
