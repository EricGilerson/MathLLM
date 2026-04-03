# Arithmetic Residual Blocks: Exact Computation Native to the Transformer Forward Pass

## 1. The Problem

Large language models cannot do arithmetic. Not reliably, and not because they lack intelligence — because their architecture has no mechanism for exact computation. A transformer predicts the next token by passing vectors through learned matrix multiplications, layer normalizations, and softmax attention. These operations are continuous, approximate, and statistical. Arithmetic is discrete, exact, and algebraic. Asking a transformer to multiply 347 × 291 is asking it to approximate a deterministic function with a probabilistic one. It sometimes gets close. It frequently does not.

The current solution is tool use: the model generates text that encodes a function call, an external system parses that text, executes the computation, serializes the result back to text, and injects it into the context window. The model then resumes generation. This works, but it is architecturally bankrupt. The autoregressive generation loop is broken. The CUDA stream stalls while control transfers to a Python process, a REST endpoint, or a subprocess. Latency spikes. Debugging becomes opaque. And the fundamental dependency on an orchestration layer — an "agent" — means the model cannot think and compute in the same cognitive step. It must stop thinking to reach for a calculator, then restart.

There is a subtler version of this failure that looks more promising but is equally flawed. Some proposals embed the tool call inside the forward pass: pause the matrix multiplications at a designated layer, extract activations to a discrete string, run an external function (a Python `eval()`, a symbolic engine), and inject the result back into the embedding space. This is a tool call in a trench coat. It still breaks the CUDA stream, still requires a discrete bottleneck, and is arguably worse because the failure mode is now hidden inside the model rather than visible in the generation trace.

The question is whether exact computation can happen natively — within the tensor flow, using standard GPU operations, without ever leaving the continuous representation space.

The answer is yes. The key is that certain algebraic structures map directly onto operations that GPUs already perform: elementwise multiplication, matrix-vector products, and trigonometric functions. If you encode numbers correctly, arithmetic becomes geometry, and geometry is what tensor hardware was built for.

## 2. The Core Insight: Arithmetic as Rotation

The Residue Number System (RNS) represents an integer not as a sequence of digits but as a tuple of remainders after division by several coprime moduli. For example, with moduli {7, 11, 13}, the number 100 is represented as (2, 1, 9) because 100 mod 7 = 2, 100 mod 11 = 1, 100 mod 13 = 9. The Chinese Remainder Theorem guarantees that this representation is unique and invertible for any integer less than the product of the moduli (7 × 11 × 13 = 1001).

The critical property of RNS is that addition and multiplication decompose into independent per-modulus operations:

$$
(a + b) \bmod p_i = ((a \bmod p_i) + (b \bmod p_i)) \bmod p_i
$$

$$
(a \times b) \bmod p_i = ((a \bmod p_i) \times (b \bmod p_i)) \bmod p_i
$$

Each modulus can be processed independently. There is no carry propagation, no sequential dependency between components. This is embarrassingly parallel — and parallelism is exactly what GPUs provide.

The second insight is geometric. Map each residue $r$ modulo $p$ to a point on the unit circle:

$$
\phi_p(r) = e^{i 2\pi r / p} = \left(\cos\frac{2\pi r}{p},\; \sin\frac{2\pi r}{p}\right)
$$

Now addition modulo $p$ becomes rotation on the circle. The complex multiplication identity guarantees:

$$
\phi_p(a) \cdot \phi_p(b) = e^{i 2\pi a/p} \cdot e^{i 2\pi b/p} = e^{i 2\pi(a+b)/p} = \phi_p(a + b)
$$

**Elementwise complex multiplication of two circle-encoded residues is exact modular addition.** No approximation. No relaxation. The result is mathematically identical to the true sum modulo $p$.

The encoding absorbs modular reduction for free: $e^{i 2\pi n/p}$ is periodic with period $p$, so $\phi_p(n) = \phi_p(n \bmod p)$ automatically. The explicit `mod` operation — which is piecewise and non-smooth — never needs to be computed. The periodicity of the circle handles it.

In real-valued coordinates (since GPUs operate on reals, not complex numbers), the operation for each prime $p_i$ is:

$$
\begin{bmatrix} \cos\frac{2\pi(a+b)}{p_i} \\[4pt] \sin\frac{2\pi(a+b)}{p_i} \end{bmatrix}
=
\begin{bmatrix}
\cos\frac{2\pi a}{p_i} \cos\frac{2\pi b}{p_i} - \sin\frac{2\pi a}{p_i} \sin\frac{2\pi b}{p_i} \\[4pt]
\sin\frac{2\pi a}{p_i} \cos\frac{2\pi b}{p_i} + \cos\frac{2\pi a}{p_i} \sin\frac{2\pi b}{p_i}
\end{bmatrix}
$$

This is a bilinear operation on four scalar inputs producing two scalar outputs — structurally identical to the gating mechanism in SwiGLU or gated attention, which are already standard in modern transformers. Four multiplications, one addition, one subtraction per prime. For $m$ primes: $6m$ FLOPs total.

Subtraction is equally simple: negate the sin component of the second operand (complex conjugation) before the same multiplication.

For integer multiplication of the underlying values, the per-modulus operation is multiplicative rather than additive. Each small prime $p_i$ has a multiplication table of at most $p_i \times p_i$ entries. For $p = 37$, the largest prime we need, that is 1,369 entries. This table is implemented as a frozen tensor: decode both residues to one-hot vectors via a frozen classifier, take their outer product to select one cell, and read out the result. The total storage for multiplication tables across all primes is roughly 15,000 parameters.

## 3. Practical Dimensionality

To handle integers up to $10^9$, select coprime moduli whose product exceeds $10^9$:

$$
p_1 = 7, \quad p_2 = 11, \quad p_3 = 13, \quad p_4 = 17, \quad p_5 = 19, \quad p_6 = 23, \quad p_7 = 29, \quad p_8 = 31, \quad p_9 = 37
$$

Their product is approximately $7.42 \times 10^9$. Nine primes.

Each prime contributes two real dimensions (cosine and sine). The full RNS circle encoding of one integer occupies 18 dimensions. Two operands require 36 dimensions. In a model with hidden dimension $d = 4096$, the computation subspace is less than 1% of the hidden state width.

The precision margin is enormous. Adjacent encodable integers are separated by at least $2\pi / p_{\max}$ radians on each unit circle. For $p_{\max} = 37$, that is approximately 9.7°. Float32 arithmetic introduces angular noise on the order of $10^{-7}$ radians, or about 0.000006°. The separation-to-noise ratio is eight orders of magnitude. Even in float16 (approximately 3 decimal digits of precision), the margin exceeds four orders of magnitude. The encoding is not approximate. It is exact, with a precision buffer so large that hardware failure is more probable than a computation error.

## 4. The Architecture

### 4.1 Base Model

A standard dense transformer. Attention layers, feedforward networks, residual connections, layer normalization. No mixture of experts. No conditional routing. No novel attention mechanisms. The model is trained and operates exactly as any conventional transformer does, with one addition: at a small number of fixed layer positions, an **Arithmetic Residual Block** (ARB) is inserted into the residual stream.

A 32-layer model might place ARBs at layers 8, 16, and 24. The first ARB is deep enough that attention has gathered operand information into individual positions. Multiple ARBs at different depths enable multi-step composition — each performs one arithmetic operation, and the standard attention layers between them propagate intermediate results forward. Three ARBs support three sequential operations, sufficient for most practical arithmetic expressions. Additional ARBs can be added at negligible cost.

### 4.2 The Arithmetic Residual Block

The ARB runs unconditionally on every token at every position, exactly like LayerNorm. There is no gating function, no routing decision, no conditional execution. It has four stages — two learned, two frozen.

```
Input: h ∈ ℝ^d (hidden state at one token position)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  STAGE 1 — EXTRACT (learned)                                │
│  Read two operands as digit vectors from hidden state       │
│                                                             │
│  STAGE 2 — ENCODE (frozen)                                  │
│  Map digit vectors to RNS circle encoding                   │
│                                                             │
│  STAGE 3 — COMPUTE (frozen)                                 │
│  Execute +, −, × in parallel via complex multiplication     │
│  Decode all results via CRT reconstruction                  │
│                                                             │
│  STAGE 4 — INJECT (learned)                                 │
│  Project results back into hidden state                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Output: h' = h + Δh ∈ ℝ^d
```

#### Stage 1: Operand Extraction (Learned)

By the time the hidden state reaches the ARB, earlier attention layers have gathered information from across the sequence. If the model is processing "347 × 291 =", the hidden state at the position of "=" contains representations of both operands, stored in separate learned subspaces. Transformers naturally develop such orthogonal "memory slots" — this is the same mechanism by which induction heads copy information with near-perfect fidelity.

Two learned linear projections extract digit-level features:

$$
\hat{d}_a = W_a h + b_a \in \mathbb{R}^K, \qquad \hat{d}_b = W_b h + b_b \in \mathbb{R}^K
$$

where $K$ is the number of digit slots. $K = 10$ covers integers up to approximately $10^{10}$. Each component targets a single digit value in $\{0, 1, \ldots, 9\}$.

These continuous outputs are then **rounded to integers**:

$$
d_a = \mathrm{round}(\hat{d}_a), \qquad d_b = \mathrm{round}(\hat{d}_b)
$$

The rounding function is non-differentiable. During training, gradients pass through via a straight-through estimator (STE): the forward pass applies `round()`, the backward pass treats it as the identity function. This is standard practice, used in quantization-aware training and vector-quantized variational autoencoders. It works because the learned projections quickly converge to near-integer outputs, minimizing the approximation introduced by the STE.

**Parameter cost.** $W_a, W_b \in \mathbb{R}^{K \times d}$. For $K = 10$, $d = 4096$: 81,920 learned parameters. This is 0.1% of a single standard transformer layer.

**Behavior on non-mathematical tokens.** The projections output arbitrary floats that round to arbitrary integers. These propagate through the frozen stages and produce meaningless results. The learned injection projection in Stage 4 learns to assign near-zero weight to these outputs. The residual connection passes $h$ through unchanged. The ARB is invisible for non-math tokens — present in the computation graph but contributing nothing to the hidden state, exactly like an attention head that attends to irrelevant content.

#### Stage 2: RNS Circle Encoding (Frozen)

Given a digit vector $d = (d_0, d_1, \ldots, d_{K-1})$ representing the integer $n = \sum_k d_k \times 10^k$, compute the weighted residue for each prime $p_i$:

$$
r_i = \sum_{k=0}^{K-1} d_k \cdot (10^k \bmod p_i)
$$

The coefficients $(10^k \bmod p_i)$ are precomputed constants. This step is a single matrix multiplication with a frozen weight matrix $C \in \mathbb{Z}^{m \times K}$:

$$
\mathbf{r} = C \, \mathbf{d}
$$

For example: $C[3, 2] = 10^2 \bmod 17 = 100 \bmod 17 = 15$.

Then map each $r_i$ to the unit circle:

$$
\phi_i = \left(\cos\frac{2\pi r_i}{p_i},\; \sin\frac{2\pi r_i}{p_i}\right)
$$

The periodicity of cosine and sine means $r_i$ does not need to be reduced modulo $p_i$ before this step. The circle encoding performs the modular reduction implicitly, keeping the entire pipeline in continuous tensor operations.

**Total encoding dimension:** $9 \text{ primes} \times 2 \text{ reals} = 18$ dimensions per operand.

#### Stage 3: Computation (Frozen)

All three operations — addition, subtraction, and multiplication — execute in parallel on every token.

**Addition.** For each prime $p_i$, elementwise complex multiplication of the two operands' circle encodings:

$$
\psi_i^{(+)} =
\begin{bmatrix}
c_{a,i} \, c_{b,i} - s_{a,i} \, s_{b,i} \\
s_{a,i} \, c_{b,i} + c_{a,i} \, s_{b,i}
\end{bmatrix}
$$

where $c_{a,i} = \cos(2\pi a / p_i)$, $s_{a,i} = \sin(2\pi a / p_i)$, etc. Nine primes, six FLOPs each: **54 FLOPs total.** A single attention head at $d = 4096$ costs on the order of 50 million FLOPs per token.

**Subtraction.** Negate the sine component of the second operand (complex conjugation), then apply the same multiplication. Cost: identical to addition plus one sign flip.

**Multiplication.** For each prime $p_i$:
1. Decode both operands' residues from their circle positions. Each point on the unit circle is one of $p_i$ evenly-spaced positions. A frozen classifier (inner product with $p_i$ template vectors, followed by argmax) identifies the residue.
2. One-hot encode both residues. Their outer product selects one cell of the multiplication table.
3. A frozen tensor $T_{p_i} \in \mathbb{R}^{p_i \times p_i}$ stores $(a \times b) \bmod p_i$ for all pairs. The outer product indexes into this table.
4. The result residue is re-encoded onto the circle for uniform representation.

For $p = 37$, the multiplication table has 1,369 entries. Across all nine primes: approximately 15,000 frozen parameters.

The model does not decide which operation to perform. It performs all three, unconditionally, and Stage 4 selects the relevant output. This follows the same design principle as multi-head attention: compute everything, let the learned projection determine relevance.

**Decoding via CRT.** The per-prime result residues $(r_1, \ldots, r_m)$ are reconstructed into the integer result using the Chinese Remainder Theorem:

$$
n = \sum_{i=1}^{m} r_i \cdot M_i \cdot (M_i^{-1} \bmod p_i) \pmod{P}
$$

where $P = \prod p_i$ and $M_i = P / p_i$. Every coefficient $M_i (M_i^{-1} \bmod p_i)$ is a precomputed constant. This is a dot product with a frozen weight vector — one inner product. The result is decomposed back into a digit vector by frozen constant-weight operations.

Three digit vectors emerge: one for the sum, one for the difference, one for the product.

#### Stage 4: Result Injection (Learned)

The three result vectors are concatenated and projected back to the hidden state dimension:

$$
\Delta h = W_{\mathrm{proj}} \cdot [\text{result}_{+} ;\; \text{result}_{-} ;\; \text{result}_{\times}] + b_{\mathrm{proj}}
$$

$W_{\mathrm{proj}} \in \mathbb{R}^{d \times 3R}$ where $R$ is the result representation dimension (approximately 30–40). About 500,000 learned parameters.

The residual connection completes the block:

$$
h' = h + \Delta h
$$

For mathematical tokens, $W_{\mathrm{proj}}$ amplifies the correct operation's result and suppresses the others. For non-mathematical tokens, $W_{\mathrm{proj}}$ outputs near-zero because training never reinforces these outputs in non-mathematical contexts. The model learns this through standard backpropagation — if the addition result helps predict the next token, the gradient strengthens the corresponding projection weights.

### 4.3 Parameter and Compute Budget

| Component | Type | Parameters | Purpose |
|---|---|---|---|
| Operand extraction ($W_a$, $W_b$) | Learned | ~80,000 | Read digit values from hidden state |
| RNS coefficient matrix ($C$) | Frozen | ~90 | Map digits to weighted residues |
| Circle encoding (cos/sin) | Frozen | 0 (functional) | Map residues to unit circle |
| Complex multiply (addition) | Frozen | 0 (functional) | Exact modular addition |
| Multiplication tables | Frozen | ~15,000 | Exact modular multiplication |
| CRT reconstruction weights | Frozen | ~80 | Recover integer from residues |
| Result projection ($W_{\mathrm{proj}}$) | Learned | ~500,000 | Inject result into hidden state |

**Per ARB:** ~15,000 frozen parameters, ~580,000 learned parameters. One standard transformer layer at $d = 4096$: ~67 million parameters. The ARB adds less than 1% parameter overhead and less than 0.001% compute overhead per layer.

## 5. Why Not a Router

The architecture described above contains no gating function, no conditional execution, and no mixture-of-experts routing. This is a deliberate design choice, not an omission.

**The cost argument.** MoE routing exists because expert networks are large — billions of parameters each — and running all experts on every token is prohibitively expensive. The ARB computation stage costs 54 FLOPs. A gating function to decide whether to skip those 54 FLOPs would itself cost more FLOPs than the computation it is gating. There is no efficiency argument for conditional execution when the module is this cheap.

**The expressiveness argument.** With unconditional execution and a learned projection, the model can partially use the ARB's output. The projection weight can be 0.3 for a "somewhat mathematical" token. A hard router forces a binary decision: routed or not routed. Soft blending through the projection is strictly more expressive.

**The reliability argument.** A router introduces a classification problem — "is this token mathematical?" — and classification problems have error rates. A misclassified mathematical token silently skips the computation. With unconditional execution, this failure mode does not exist. The ARB always runs; the projection always has the opportunity to use its output.

**The training argument.** MoE gating requires auxiliary losses to prevent routing collapse, Gumbel-softmax or straight-through estimators for the discrete routing decision, and careful load-balancing hyperparameters. The unconditional ARB trains like any other residual block — standard backpropagation through the learned projection, no special treatment.

The ARB is not an "expert" in the MoE sense. It is a **layer** — a structural component of the architecture, as fundamental as LayerNorm or the residual connection. LayerNorm is not gated by a router. The residual connection is not conditionally skipped. The ARB belongs in the same category.

## 6. Floating-Point Numbers

The frozen computation stages operate on integers. Floating-point numbers are handled by representing them as scaled integers — a fixed-point encoding managed entirely by the learned extraction stage.

The number $3.14159$ becomes the pair $(\text{mantissa} = 3141590,\; \text{exponent} = -6)$. Both are integers. The extraction stage (Stage 1) learns to decompose the hidden state into mantissa digits and exponent digits rather than a single digit vector. The frozen stages see only integers and operate exactly as before.

**Addition and subtraction.** Align exponents, then add mantissas:

$$
(a \times 10^p) + (b \times 10^q) = (a \times 10^{p-q} + b) \times 10^q \qquad (p \geq q)
$$

Exponent alignment requires one integer multiplication ($a \times 10^{p-q}$); the addition is a second integer operation. Two sequential ARB invocations across two layers. Exact.

**Multiplication.**

$$
(a \times 10^p) \times (b \times 10^q) = (a \times b) \times 10^{p+q}
$$

Mantissa multiplication and exponent addition. Both integer operations. One ARB layer. Exact.

**Precision.** Working with $P = 6$ decimal places (mantissas scaled by $10^6$) provides TI-84-grade accuracy. The TI-84 uses 14 internal digits; pushing to $P = 9$ requires only expanding the RNS modulus set — more or slightly larger primes — with no structural change to the architecture.

**Chained operations.** Fixed-point arithmetic does not accumulate rounding error the way IEEE 754 floating-point does. Each intermediate result is exact to $P$ digits. Rounding occurs only once, at final representation. For multi-step calculations, this is strictly better than standard floating-point, where every intermediate operation introduces a fresh rounding error.

## 7. Division

Division is the one operation that does not map cleanly onto RNS. Addition and multiplication are componentwise across modular rings. Division is not — integer division with remainder requires magnitude comparison, and RNS does not preserve ordering information. You cannot determine which of two RNS-encoded numbers is larger without first converting back to standard representation.

Modular inverse exists (each $p_i$ is prime, so $b^{-1} \bmod p_i$ is well-defined for $b \neq 0$), but modular division and integer division are fundamentally different operations. If $b$ does not exactly divide $a$, computing $a \times b^{-1} \bmod p_i$ produces a result that wraps around the modular ring — not the quotient, not an approximation, but an unrelated number. For exact division (12 ÷ 4 = 3), the ARB handles it natively via modular inverse. For general division (7 ÷ 3), it cannot.

The clean solution uses the operations that do work — multiplication and subtraction — to converge on division iteratively.

### Newton-Raphson Reciprocal

The Newton-Raphson iteration for $1/b$ is:

$$
x_{n+1} = x_n \cdot (2 - b \cdot x_n)
$$

Each iteration requires two multiplications and one subtraction — all operations the ARB executes exactly. Convergence is quadratic: the number of correct digits doubles with each iteration.

The design uses collaboration between the learned and frozen components:

1. **The transformer's FFN layers** (learned, approximate) produce an initial estimate $x_0 \approx 1/b$. Even 1–2 digits of accuracy suffice. Rough estimation is something neural networks are good at.

2. **ARB at layer 8** computes $x_1 = x_0(2 - b x_0)$. If $x_0$ had 2 correct digits, $x_1$ has 4.

3. **ARB at layer 16** computes $x_2 = x_1(2 - b x_1)$. Now 8 correct digits.

4. **ARB at layer 24** computes $x_3 = x_2(2 - b x_2)$. Now 16 correct digits — full float64 precision from three iterations.

5. **Final multiplication:** $a / b = a \times (1/b)$. One more ARB invocation.

The model does not need to learn division. It needs to learn approximately what $1/b$ is — which it already can — and the frozen ARB layers mechanically correct whatever error remains. Each component does what it is best at: the neural network estimates, the algebraic circuit refines.

### Edge Cases

**Irrational results** ($\sqrt{2}$, $\pi$, etc.) cannot be represented exactly in any finite-precision system. The ARB provides the same answer a handheld calculator provides: correct to $P$ digits. No calculator gives exact $\sqrt{2}$ either.

**Overflow.** If a mantissa product exceeds the RNS representable range (~$10^9$ with the nine-prime basis), CRT reconstruction silently wraps around. Mitigations: expand the prime set (increasing range at negligible parameter cost), or train the extraction stage to normalize operand magnitudes before computation.

## 8. Exponentiation

Integer exponentiation $a^b$ appears to require repeated multiplication — $b$ sequential ARB invocations. For large $b$, this exceeds the available network depth. Exponentiation by squaring reduces the cost to $\log_2(b)$ multiplications, but even this consumes many ARB layers for large exponents.

RNS offers a far better path. The key is Fermat's Little Theorem: for prime $p$ and $a$ not divisible by $p$,

$$
a^{p-1} \equiv 1 \pmod{p}
$$

This means:

$$
a^b \bmod p = a^{(b \bmod (p-1))} \bmod p
$$

For our largest prime $p = 37$, the effective exponent is always at most 35, regardless of how large $b$ actually is. $a^{1000000} \bmod 37 = a^{(1000000 \bmod 36)} \bmod 37 = a^{(16)} \bmod 37$. The astronomically large exponent collapses to a small number.

This makes exponentiation a **frozen lookup table**, identical in structure to the multiplication tables already in Stage 3.

### Implementation

For each prime $p_i$, precompute and store a frozen exponentiation table:

$$
T_{\mathrm{exp}}^{(p_i)}[a, k] = a^k \bmod p_i \qquad \text{for } a \in \{0, \ldots, p_i - 1\},\; k \in \{0, \ldots, p_i - 2\}
$$

For $p = 37$, this table has $37 \times 36 = 1{,}332$ entries. Across all nine primes, approximately 10,000 frozen parameters — the same order as the multiplication tables.

The computation flow within a single ARB layer:

1. **Extract** digits of $a$ and $b$ from the hidden state (learned, same as before).
2. **Encode base:** compute $a \bmod p_i$ for each prime via the existing frozen coefficient matrix $C$.
3. **Encode exponent:** compute $b \bmod (p_i - 1)$ for each prime via a second frozen coefficient matrix $C_{\mathrm{exp}} \in \mathbb{Z}^{m \times K}$, where $C_{\mathrm{exp}}[i, k] = 10^k \bmod (p_i - 1)$. These moduli are $\{6, 10, 12, 16, 18, 22, 28, 30, 36\}$. Same structure, same cost — one matrix multiply with ~90 frozen parameters.
4. **Table lookup:** index into $T_{\mathrm{exp}}^{(p_i)}$ with the base residue and exponent residue. Same mechanism as multiplication: one-hot encode both indices, outer product selects the cell, frozen tensor returns the result.
5. **CRT reconstruct** the integer result (existing frozen path).

**One ARB layer. One pass. Arbitrarily large exponents.** The Fermat reduction is algebraically exact — it is not an approximation. The only limitation is that the *result* $a^b$ must fall within the representable range (~$10^9$). For context, $2^{30} \approx 10^9$, so exponents of common bases are well-covered.

Adding exponentiation to the ARB requires extending Stage 3 with one additional frozen coefficient matrix (~90 parameters) and one set of frozen lookup tables (~10K parameters). The Stage 1 extraction and Stage 4 injection remain unchanged — the exponentiation result is simply a fourth output alongside addition, subtraction, and multiplication, selected by the same learned projection.

## 9. Trigonometric Functions

Trigonometric functions are transcendental — $\sin(x)$ is irrational for almost all rational $x$. No finite system computes them exactly. A TI-84 does not compute exact $\sin$; it evaluates a polynomial approximation to 14 digits. The ARB can do the same, with one structural advantage: every step of the polynomial evaluation uses exact integer arithmetic, so the **only source of error is the polynomial truncation itself**, not accumulated floating-point imprecision.

### The Approach: Frozen Polynomial Evaluation

Any smooth function on a bounded interval can be approximated to arbitrary precision by a polynomial. For trigonometric functions, the optimal polynomials (minimax or Chebyshev) are extremely well-studied and have been used in every math library since the 1960s.

For $\sin(x)$ with $x \in [-\pi/4, \pi/4]$, a degree-7 minimax polynomial achieves better than 10 digits of accuracy:

$$
\sin(x) \approx c_1 x + c_3 x^3 + c_5 x^5 + c_7 x^7
$$

where $c_1, c_3, c_5, c_7$ are known constants (e.g., $c_1 \approx 1$, $c_3 \approx -1/6$, $c_5 \approx 1/120$, $c_7 \approx -1/5040$, though minimax coefficients differ slightly from Taylor for optimal worst-case error).

This polynomial is evaluated using Horner's method — rewriting it as a chain of multiply-accumulate operations, innermost first:

$$
\sin(x) \approx x \cdot \left(c_1 + x^2 \cdot \left(c_3 + x^2 \cdot \left(c_5 + c_7 \cdot x^2\right)\right)\right)
$$

Let $u = x^2$. The evaluation unfolds as:

| Step | Operation | ARB Layer |
|------|-----------|-----------|
| 1 | $u = x \times x$ | Layer $L$ |
| 2 | $t_1 = c_7 \times u + c_5$ | Layer $L+1$ |
| 3 | $t_2 = t_1 \times u + c_3$ | Layer $L+2$ |
| 4 | $t_3 = t_2 \times u + c_1$ | Layer $L+3$ |
| 5 | $\sin(x) = t_3 \times x$ | Layer $L+4$ |

Five ARB layers for a degree-7 polynomial evaluation to ~10 digits. Each step is a multiply (exact in the ARB) and an add (exact in the ARB). The polynomial coefficients $c_1, \ldots, c_7$ are frozen constants embedded in the extraction stage — one operand is the running accumulator $t_k$ from the previous layer, the other is either $u$ or the next coefficient.

For $\cos(x)$ on the same interval, an even-degree polynomial is used:

$$
\cos(x) \approx 1 + c_2 x^2 + c_4 x^4 + c_6 x^6
$$

Same Horner structure, same number of ARB layers, different frozen coefficients.

### Range Reduction

The polynomial is accurate only on $[-\pi/4, \pi/4]$. For arbitrary $x$, range reduction maps it into this interval using trigonometric identities.

**Step 1: Reduce modulo $2\pi$.** Multiply $x$ by the frozen constant $1/(2\pi)$ (stored as a fixed-point integer). The integer part gives the number of full periods; the fractional part is $x \bmod 2\pi$ (scaled). This is one ARB multiplication. The fractional-part extraction uses the same round-and-subtract mechanism as the digit rounding in Stage 1.

**Step 2: Determine the octant.** The reduced value falls in one of eight octants of the unit circle. Which octant determines the sign of the result and whether to evaluate the sine or cosine polynomial. This is a comparison against frozen constants ($\pi/4$, $\pi/2$, etc.), implementable via the learned extraction stage — the model learns to read the octant from the reduced value and route accordingly through the injection projection.

**Step 3: Evaluate the polynomial** on the final reduced argument $r \in [0, \pi/4]$ using the Horner chain above.

Total ARB layers for a trigonometric evaluation: 1 (range reduction) + 5 (polynomial) = **6 ARB layers**.

### Practical Placement

A 32-layer model with ARBs at layers 4, 8, 12, 16, 20, 24, and 28 provides:

- **Layers 4–8:** basic arithmetic (addition, subtraction, multiplication, exponentiation).
- **Layers 8–24:** trig polynomial evaluation (Horner chain across 5 sequential ARBs), also available for multi-step arithmetic composition.
- **Layer 28:** final operations and result composition.

Seven ARBs at ~580K learned parameters each: 4.1M total learned parameters. Still a small fraction of any base model. The frozen parameter cost per ARB increases slightly (adding the exponentiation tables and polynomial coefficients) but remains under 30K per block.

### Why the Error Is Controlled

Every step of the Horner evaluation uses exact integer arithmetic — the ARB's addition and multiplication are mathematically precise within the representable range. No step introduces floating-point rounding error. The only error is from truncating the polynomial series: a degree-7 minimax polynomial for $\sin(x)$ on $[-\pi/4, \pi/4]$ has a worst-case error of approximately $10^{-10}$.

This is fundamentally different from how CPUs evaluate trig functions. A CPU performs each multiply-add in IEEE 754 floating point, introducing a fresh rounding error at every step. After 5–6 chained operations, these errors compound. The ARB chain does not compound errors because each intermediate result is exact — the multiplication of two $P$-digit fixed-point numbers produces a $2P$-digit exact result, and the fixed-point scaling preserves all digits through the chain.

The implication: for the same polynomial degree, the ARB evaluation is **more accurate** than an equivalent CPU evaluation, because it eliminates computational error entirely and leaves only approximation error.

### Extension to Other Transcendental Functions

The same polynomial-evaluation pattern handles any function with a known minimax approximation on a bounded interval:

- **$\cos(x)$**: even-degree polynomial, same Horner structure, same ARB cost.
- **$\tan(x)$**: compute as $\sin(x) / \cos(x)$ using the sine and cosine polynomials plus one Newton-Raphson division.
- **$\exp(x)$**: range reduce via $x = k \ln 2 + r$ (one multiply by $1/\ln 2$, extract integer $k$ and remainder $r$). Evaluate $e^r$ by polynomial for $r \in [0, \ln 2]$. Multiply by $2^k$ (an exact power-of-two scaling).
- **$\ln(x)$**: represent $x = m \times 2^e$ (extract mantissa and exponent). Evaluate $\ln(m)$ by polynomial for $m \in [1, 2)$. Add $e \ln 2$ (a frozen constant multiplication).
- **$\sqrt{x}$**: Newton-Raphson iteration $x_{n+1} = (x_n + a/x_n) / 2$, converging quadratically. Each iteration is one division (itself Newton-Raphson) and one addition. Two to three iterations from a rough initial estimate give full precision.

Each follows the same template: range reduction (one ARB layer using exact multiplication), polynomial or iterative evaluation (several ARB layers), and CRT reconstruction. The frozen polynomial coefficients are different for each function but the architectural pattern is identical.

## 10. Training: Retrofitting an Existing Model


### Overview

The ARB is designed to be surgically inserted into a pretrained open-source model — not trained from scratch. No base model pretraining is required. The existing model's weights are kept entirely frozen; only the ARB's small learned interface (~580K parameters per block) is trained. This makes validation feasible on a single consumer GPU with a small model.

### Step 1: Select a Base Model

Choose a small, well-understood open-source transformer where the architecture internals are fully accessible. Suitable candidates:

- **GPT-2 (124M)** — simple, well-documented, trivial to modify in HuggingFace Transformers. 12 layers, $d = 768$. Place ARBs at layers 4, 8, and 10.
- **Pythia (70M–410M)** — EleutherAI's suite with consistent architecture across scales. Good for ablation studies comparing ARB effectiveness at different model sizes.
- **LLaMA 3.2 (1B)** — if more capacity is needed, but the 124M–410M range is sufficient to prove the concept.

The base model must use a standard dense transformer architecture (not MoE) and a tokenizer that emits individual digit tokens. Most do — GPT-2 tokenizes "347" as separate digit tokens in many contexts, and Pythia and LLaMA handle digits similarly.

### Step 2: Surgical Insertion

Modify the model's forward pass to insert ARB layers at the chosen positions. In PyTorch, this means wrapping the existing transformer blocks:

1. **Freeze all base model parameters.** Set `requires_grad = False` on every existing parameter. The pretrained weights do not change at any point.
2. **Construct the frozen ARB stages.** Initialize the RNS coefficient matrix $C$, the cos/sin encoding functions, the multiplication table tensors, and the CRT reconstruction weights. Register these as non-trainable buffers (`register_buffer`).
3. **Initialize the learned ARB stages.** Create $W_a$, $W_b$ (operand extraction) and $W_{\mathrm{proj}}$ (result injection) as new `nn.Parameter` objects with `requires_grad = True`. **Initialize $W_{\mathrm{proj}}$ to zero** so that the ARB starts as a no-op — the residual connection passes the hidden state through unchanged, and the model behaves identically to the unmodified base.
4. **Insert each ARB as an `nn.Module`** between the chosen transformer layers. The forward pass becomes: attention → FFN → ARB → next layer.

At initialization, the modified model produces exactly the same outputs as the original. The ARBs exist in the computation graph but contribute nothing until their learned parameters are trained.

### Step 3: Fine-Tuning the Interface

Train only the ARB's learned parameters ($W_a$, $W_b$, $b_a$, $b_b$, $W_{\mathrm{proj}}$, $b_{\mathrm{proj}}$) on arithmetic data. Everything else — the base model's attention weights, FFN weights, embeddings, layer norms — stays frozen.

**Training data.** The entire dataset is synthetically generated — a Python script with `random.randint` and standard math functions produces unlimited labeled examples with guaranteed correct answers. No human annotation or curated datasets are required.

The critical design principle is **format diversity, not just quantity.** If every training example is `"347 + 291 = 638"`, the extraction layer overfits to that exact syntactic pattern and fails the moment the format changes. The learned projection needs to discover when the ARB's output is useful based on the hidden state's semantic content, not surface-level pattern matching on token sequences. This requires three categories of training data:

**Category 1: Positive examples across varied formats.** For each sampled (operation, operands, result) tuple, render through a randomly chosen template. 50–100 templates per operation ensures the model cannot memorize syntax:

- Direct: `347 + 291 = 638`, `347+291=638`, `347 + 291 → 638`
- Natural language: `"the sum of 347 and 291 is 638"`, `"adding 347 to 291 gives 638"`, `"347 and 291 combined make 638"`
- Questions: `"What is 347 plus 291? 638"`, `"Calculate 347 + 291. The answer is 638."`
- LaTeX: `$347 + 291 = 638$`, `\( 347 + 291 = 638 \)`
- Code: `x = 347 + 291  # 638`, `result = add(347, 291)  // returns 638`
- Tables: `| 347 | 291 | 638 |`
- Separated operands: `"I have 347 apples. My friend gives me 291 more. Now I have 638 apples."`
- Reasoning traces: `"...so we need 347 + 291, which is 638, and then..."`, `"Let me calculate: 347 + 291. That gives us 638."`
- Multi-step with intermediate results: `"First, 3 + 4 = 7. Then, 7 × 5 = 35."`
- Trig and transcendentals: `"sin(1.047) ≈ 0.866025"`, `"$e^{2.5} = 12.182494$"`, `"cos(π/3) = 0.5"`

**Category 2: Negative examples (numbers present, no computation needed).** Roughly equal in volume to positive examples. Sampled from real text corpora (WikiText, code repositories, news articles) containing numbers that are not being computed on:

- `"Flight 347 departs from gate 291 at terminal 3."`
- `"The population in 2019 was 347,291."`
- `"See section 3.4.7 on page 291."`
- `"Model GPT-347 was trained on 291 billion tokens."`
- Code with numeric constants: `MAX_RETRIES = 347`, `port = 8291`

Without negative examples, the projection learns "numbers visible → inject ARB output" and hallucinates arithmetic into non-math contexts. The negative examples train the projection to output near-zero when the hidden state contains numbers but no computation is expected.

**Category 3: Edge cases and boundary conditions.**

- Operands spanning all digit counts from 1 to 10, uniformly sampled (not biased toward small numbers).
- Operands near the representable range boundary (~$10^9$).
- Zero as an operand: `"347 + 0 = 347"`, `"0 × 291 = 0"`.
- Identity operations: `"347 × 1 = 347"`, `"347 - 347 = 0"`.
- Trig edge cases: `sin(0) = 0`, `cos(0) = 1`, `sin(π/2) = 1`.

**Generator structure.** A single Python script with two halves: one samples (operation, operands) and renders through a random template; the other samples real text containing numbers from a corpus. Mix approximately 50/50. The loss function is standard next-token cross-entropy on the full sequence — the model learns organically when the ARB output helps predict the next token and when it does not.

A training set of 100K–200K examples (roughly 50K positive across varied templates, 50K negative from corpus text, plus edge cases) should be sufficient given the small number of learned parameters (~1.7M total for three ARBs).

**Optimization.** Standard AdamW on the ARB parameters only. Learning rate in the 1e-4 to 1e-3 range — higher than typical fine-tuning because the learned surface is small and the target behavior is well-defined. The `round()` in Stage 1 uses a straight-through estimator for gradient flow.

**What the model learns during this phase:**
- The extraction projections ($W_a$, $W_b$) learn which subspaces of the hidden state contain operand digit information — aligning with wherever the pretrained attention layers already store gathered number representations.
- The injection projection ($W_{\mathrm{proj}}$) learns to amplify the correct operation's result for math tokens and suppress all ARB output for non-math tokens.

**What the model does not need to learn:** Arithmetic itself. Number encoding. Modular reduction. CRT reconstruction. These are frozen and correct by construction.

### Step 4: Evaluation

Compare the modified model against the unmodified base on arithmetic benchmarks:

- **Exact-match accuracy** on $N$-digit addition, subtraction, and multiplication for $N = 1, \ldots, 10$.
- **Ablation:** set $W_{\mathrm{proj}}$ back to zero and confirm accuracy reverts to baseline — proving the ARB is the source of improvement, not incidental fine-tuning effects.
- **Non-math regression:** verify that perplexity on standard language benchmarks (WikiText, LAMBADA) is unchanged. Since the base model is frozen, the ARB cannot degrade language capability.
- **Division accuracy** on expressions requiring Newton-Raphson iteration across multiple ARB layers.

The hypothesis to validate: the modified model achieves near-perfect arithmetic accuracy on in-range integers while the unmodified base model shows the typical degradation on multi-digit operations. If this holds, the ARB is working — the model has learned to route its existing number representations through the frozen computation path and use the exact results.

### Why This Is Tractable

The total learned parameter count for three ARBs is approximately 1.7 million — about 1.4% of GPT-2's 124M parameters and a vanishing fraction of larger models. The base model is frozen, so there is no catastrophic forgetting, no need for replay buffers, and no risk of degrading existing capabilities. The training data is synthetic and unlimited. The entire experiment — model surgery, data generation, fine-tuning, evaluation — is feasible on a single consumer GPU (an RTX 3090 or equivalent) in hours, not days.

The frozen stages never receive gradients, cannot drift, cannot degrade under distribution shift, and cannot develop adversarial failure modes. The only learned failure mode is imprecise operand extraction — and the `round()` + STE mechanism actively fights this by rewarding the extraction weights for producing near-integer outputs.

## 11. Multi-Step Composition

For an expression like $(3 + 4) \times 5$, two sequential operations are required. This maps to network depth:

1. **ARB at layer 8** computes $3 + 4 = 7$. The result is injected into the hidden state.
2. **Attention layers 9–15** propagate the intermediate result. The hidden state at the relevant position now carries both the intermediate result (7) and the second operand (5).
3. **ARB at layer 16** computes $7 \times 5 = 35$.

Three ARBs support three sequential operations. For longer chains, the model decomposes the computation across generation steps — emitting intermediate results as tokens, then using them in subsequent forward passes. This is analogous to a human writing intermediate results on paper, and it arises naturally from the architecture without any special mechanism.

## 12. Why This Design

**Why RNS.** The Residue Number System converts one large-integer operation into $m$ independent small-modulus operations. Each is exact in float32. The Chinese Remainder Theorem guarantees exact reconstruction. Alternative encodings — binary, positional decimal — require carry propagation, which is inherently sequential and difficult to express in a single frozen matrix operation. RNS is embarrassingly parallel.

**Why circle encoding.** Mapping residues to unit circles makes modular addition a rotation — a complex multiplication, which is a standard bilinear tensor operation. It eliminates explicit `mod` computation via the natural periodicity of trigonometric functions. And the angular separation between encodable values provides the precision guarantee: eight orders of magnitude of margin in float32.

**Why frozen weights.** Learned parameters drift, degrade under distribution shift, and develop unpredictable failure modes. The frozen computation stages are mathematically proven correct for all valid inputs within the representable range. Exactness is a property of the algebra, not a property of the training run. The learned components handle only the interface — reading operands from the model's representation, writing results back — while the computation itself is immutable.

**Why unconditional execution.** Running all operations on every token eliminates routing errors, requires no auxiliary losses, and allows the model to partially or fully use results through soft projection weights. The negligible computational cost (~54 FLOPs versus ~50 million FLOPs per attention head) makes conditional execution an unnecessary complication.

**Why a standard dense transformer.** The ARB is structurally identical to any other residual block. It operates on single positions, produces a fixed-dimensional output, and connects via a residual connection. It does not require architectural modifications to attention, feedforward networks, positional encoding, or the output head. It can be inserted into any existing transformer architecture as a drop-in layer.

## 13. Beyond Arithmetic

The ARB is a proof of concept for a more general principle: **any bounded deterministic function that can be expressed as a composition of standard tensor operations can be embedded as a frozen residual block.**

The pattern is:
1. Choose a structured encoding where the target operation becomes a simple tensor operation (elementwise multiply, matrix-vector product, convolution).
2. Implement the encoding, computation, and decoding as frozen weight matrices.
3. Wrap it in a learned extraction and injection layer.
4. Insert it as a residual block.

Candidates beyond arithmetic:
- **String matching.** Convolution-based pattern matching with frozen filter banks.
- **Sorting.** Permutation matrices constructed from frozen comparator networks.
- **Symbolic binding.** Circular convolution for variable binding and unbinding, as in Vector Symbolic Architectures.
- **Unit conversion.** Frozen multiplication by known physical constants.
- **Date arithmetic.** Modular operations on day/month/year representations.

Each of these is bounded (finite input, finite output, guaranteed termination), deterministic, and expressible in tensor operations. Each could be a frozen residual block costing negligible compute and providing exact results.

If even one such module — the arithmetic ARB — is shown to work, it validates the architectural pattern. The implication is that tool use does not require agents, orchestration layers, or generation loop interruptions. It requires the right encoding and a frozen layer.

## 14. What Could Go Wrong

**Operand gathering.** The ARB assumes that by the time the hidden state reaches Stage 1, attention has gathered precise operand information into the current position. Attention is fundamentally a soft-weighted sum, not a hard copy. For simple cases (two operands, moderate context distance), induction-head-like mechanisms provide near-perfect copying. For complex cases (many operands scattered across long contexts, nested expressions), the precision of the gathered representation may degrade. This is the most likely practical failure mode — not wrong computation, but misread operands.

**Operand identification.** The extraction projections must learn which subspaces of the hidden state contain "operand A" versus "operand B." This is a representation alignment problem between the attention mechanism (which decides where to put information) and the extraction layer (which decides where to read it from). It should converge during Phase 2 training, but pathological cases may exist.

**Range limitations.** The nine-prime basis supports integers up to ~$10^9$. For larger numbers, add more primes. Each additional prime costs 2 dimensions in the encoding and one more row in the coefficient matrix. The architecture scales gracefully, but the choice of primes must be made at model design time, not at inference time. The representable range is a fixed architectural parameter.

**Operations not yet implemented.** Logarithms and square roots are addressed architecturally (Section 9) but not yet validated. Higher-order functions (e.g., Bessel functions, gamma function) would require longer polynomial chains and more ARB layers, potentially exceeding practical depth limits for single-pass evaluation.

## 15. Summary

The Arithmetic Residual Block embeds exact computation into the transformer forward pass as a frozen residual layer. Numbers are encoded in the Residue Number System on unit circles, where addition is complex multiplication and the Chinese Remainder Theorem guarantees exact reconstruction. The computation occupies 36 dimensions of the hidden state, costs 54 FLOPs for addition, and adds less than 1% parameter overhead.

The architecture covers the full range of calculator operations. Addition, subtraction, and multiplication are exact in a single ARB layer. Exponentiation exploits Fermat's Little Theorem to collapse arbitrarily large exponents into frozen lookup tables — also a single layer. Division uses Newton-Raphson iteration across multiple ARB layers, with the transformer providing initial estimates and the frozen layers refining them to full precision. Trigonometric and other transcendental functions are evaluated via Horner's method on frozen minimax polynomials, where every arithmetic step is exact and the only error is the polynomial truncation — yielding better accuracy than equivalent CPU floating-point chains. Floating-point numbers throughout are supported via fixed-point mantissa-exponent encoding.

Each block runs unconditionally on every token, requires no routing or gating, and trains via standard backpropagation on a small learned interface (~580K parameters per block). The frozen computation stages cannot drift, degrade, or fail — their correctness is a mathematical property, not an empirical one. The entire system can be validated by surgically inserting ARBs into a pretrained open-source model and fine-tuning only the learned interface on synthetic data — a single-GPU experiment.

If validated, this architecture demonstrates that exact deterministic computation and probabilistic language generation can coexist in the same tensor flow, without agents, without tool calls, and without ever leaving the GPU.
