# MathLLM

**Arithmetic Residual Blocks — Exact Computation Native to the Transformer Forward Pass**

*Don't Learn What You Can Compute.*

MathLLM implements Arithmetic Residual Blocks (ARBs): frozen, differentiable modules inserted into a transformer's forward pass that compute exact integer arithmetic using Residue Number System (RNS) encoding on unit circles. The base model stays completely frozen — only a thin learned interface (1.7M parameters, 0.47% of the model) is trained. No tool calls, no CUDA stream breaks, no external calculators. Arithmetic happens inside the forward pass as standard tensor operations.

On SmolLM2-360M with operands up to 3 digits: **99.95% exact-match accuracy** across addition, subtraction, multiplication, and division — up from 4.6% for the unmodified model.

---

## How It Works

### RNS Circle Encoding

Integers are represented not as digit sequences but as tuples of remainders modulo several coprime primes. For example, with primes {7, 11, 13}, the number 100 becomes (2, 1, 9) — because 100 mod 7 = 2, 100 mod 11 = 1, 100 mod 13 = 9. The Chinese Remainder Theorem guarantees this representation is unique and invertible up to the product of the primes.

Each residue is then mapped to a point on a unit circle:

```
φ_p(r) = (cos(2πr/p), sin(2πr/p))
```

This encoding converts arithmetic into geometry:

- **Addition** becomes complex multiplication (rotation): `φ(a) · φ(b) = φ(a+b)`. Six FLOPs per prime, mathematically exact.
- **Subtraction** is conjugation followed by the same rotation.
- **Multiplication** uses frozen lookup tables: decode both residues, index a precomputed `p × p` table, re-encode.
- **Division** computes the modular inverse (every prime admits one), then a single table lookup.
- **Exponentiation** exploits Fermat's Little Theorem to collapse arbitrarily large exponents into a frozen lookup.

The precision margin is enormous. Adjacent residues are separated by ~15.7° on the circle. Float32 noise is ~0.000006°. That's **8 orders of magnitude** of safety margin — the encoding is exact within hardware precision.

We use six primes {7, 11, 13, 17, 19, 23} with product ~3.2M, covering 3-digit operands. The prime set is extensible — adding {29, 31, 37} covers up to ~7.4 × 10⁹ at negligible cost.

### The Four-Stage Pipeline

Each ARB is a four-stage pipeline. Stages 1–3 form the **compute core** (runs once per forward pass); Stage 4 is the **injection layer** (instantiated at each ARB position).

```
Hidden State h
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: EXTRACT (frozen, deterministic)               │
│  Scan tokens for operators (+, -, ×, ÷)                 │
│  Collect digit tokens → digit vectors d_a, d_b          │
├─────────────────────────────────────────────────────────┤
│  Stage 2: ENCODE (frozen, precomputed constants)        │
│  Digit vectors → RNS residues → unit circle points      │
│  12 dimensions per operand (6 primes × 2 reals)         │
├─────────────────────────────────────────────────────────┤
│  Stage 3: COMPUTE (frozen, lookup tables)               │
│  +, −, ×, ÷, exp in parallel across all primes          │
│  CRT reconstruction → integer result → digit vector     │
├─────────────────────────────────────────────────────────┤
│  Stage 4: INJECT (learned, gated)                       │
│  h' = h + σ(g) · MLP(result)                            │
│  Fires only at = and subsequent answer digit positions   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Hidden State h' → LM Head (+ LoRA)
```

**Deterministic extraction** is a deliberate design choice: the ARB is a calculator, and its interface is the equation. The model invokes the ARB by *writing* — generating `A * B =` is the tool use. Every invocation is visible in the generated text, interpretable by construction.

**Unconditional execution**: all operations run on every token. The injection gate learns relevance. A router to skip 36 FLOPs would cost more than the computation it gates.

**Gated injection**: the gate initializes open (σ(3.0) ≈ 0.95) — the model starts with high trust in the ARB and training confirms it.

## Results

Training on SmolLM2-360M (360M parameters frozen, 1.7M trained) with 250K synthetic arithmetic examples:

| Digits | Add (Base) | Add (ARB) | Sub (Base) | Sub (ARB) | Mul (Base) | Mul (ARB) | Div (Base) | Div (ARB) |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1 × 1  | 17.3 | **100.0** | 0.9  | **100.0** | 24.9 | **100.0** | 9.5  | **100.0** |
| 1 × 2  | 1.5  | **100.0** | 0.8  | **100.0** | 5.4  | **100.0** | 1.9  | **100.0** |
| 1 × 3  | 0.9  | **100.0** | 0.7  | **99.5**  | 1.7  | **100.0** | 1.0  | **100.0** |
| 2 × 1  | 18.9 | **100.0** | 0.4  | **100.0** | 25.5 | **100.0** | 1.1  | **100.0** |
| 2 × 2  | 7.9  | **100.0** | 1.0  | **100.0** | 3.5  | **100.0** | 0.0  | **100.0** |
| 2 × 3  | 0.8  | **100.0** | 0.0  | **100.0** | 0.9  | **99.9**  | 0.0  | **99.9**  |
| 3 × 1  | 8.3  | **100.0** | 1.3  | **99.6**  | 17.2 | **99.9**  | 1.8  | **100.0** |
| 3 × 2  | 3.9  | **100.0** | 0.0  | **100.0** | 0.8  | **99.9**  | 0.0  | **100.0** |
| 3 × 3  | 4.2  | **99.9**  | 0.1  | **100.0** | 0.0  | **99.7**  | 0.1  | **100.0** |
| **Mean** | 7.1 | **99.99** | 0.6 | **99.90** | 8.9 | **99.93** | 1.7 | **99.99** |

**Grand mean**: Base 4.6% → ARB **99.95%** (36,000 exact-match examples, 1,000 per digit-pair per operation).

Only 17 errors in 36,000 examples. Division achieves the *highest* accuracy (99.99%) despite being the most complex operation — because the base model has the weakest prior on division (1.7% baseline) and offers the least resistance to the injection signal. This reveals that residual errors are interference from the frozen model's competing approximate circuits, not limitations of the ARB computation. It predicts that pre-training with ARBs — where no competing prior exists — would yield ceiling performance across all operations.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Generate training data (250K synthetic arithmetic examples)
python scripts/generate_data.py --config configs/360m.yaml

# Train (auto-resumes from checkpoints)
python scripts/train.py --config configs/360m.yaml

# Train in increments
python scripts/train.py --config configs/360m.yaml --epochs-to-run 10
python scripts/train.py --config configs/360m.yaml --steps-to-run 1000

# Export a standalone model bundle from a checkpoint
python scripts/export_checkpoint.py --config configs/360m.yaml

# Evaluate
python scripts/evaluate.py --config configs/360m.yaml
python scripts/evaluate.py --model-dir trained_model/

# Zero-forgetting benchmark (WikiText-103 + PIQA + HellaSwag)
python scripts/benchmark_zero_forgetting.py --config configs/360m.yaml
python scripts/benchmark_zero_forgetting.py --model-dir trained_model_360m/

# Inference
python scripts/infer.py --model-dir trained_model/ --prompt "347 * 291 ="
python scripts/infer.py --model-dir trained_model/  # interactive REPL
```

## Project Structure

```
mathllm/
├── arb/                        # Arithmetic Residual Block
│   ├── arb_module.py           #   Compute core (stages 1-3) and injector (stage 4)
│   ├── stage1_extract.py       #   Deterministic operand extraction from tokens
│   ├── stage2_encode.py        #   Frozen RNS circle encoding
│   ├── stage3_compute.py       #   Frozen arithmetic (+, −, ×, ÷, exp)
│   ├── stage4_inject.py        #   Learned gated injection
│   ├── constants.py            #   Precomputed RNS constants and lookup tables
│   └── ste.py                  #   Straight-Through Estimator for gradient flow
├── data/
│   ├── generator.py            #   Synthetic arithmetic data generation
│   ├── dataset.py              #   PyTorch Dataset with tokenization
│   ├── templates.py            #   1100+ natural-language format templates
│   └── negative_examples.py    #   Non-arithmetic text with numbers
├── model/
│   ├── gpt2_arb.py             #   TransformerWithARB (frozen base + ARB insertion)
│   ├── lora.py                 #   LoRA adapter for LM head
│   └── utils.py                #   Model utilities (freeze, device handling)
├── training/
│   ├── trainer.py              #   Training loop with checkpointing & early stopping
│   └── losses.py               #   Answer-only loss masking
├── evaluation/
│   └── evaluator.py            #   Exact-match accuracy by operation & digit count
└── config.py                   #   Nested dataclass config with YAML loading

configs/
└── 360m.yaml                   #   SmolLM2-360M configuration

scripts/
├── generate_data.py            #   Generate synthetic training data
├── train.py                    #   Train with checkpointing & auto-resume
├── export_checkpoint.py        #   Export model bundle from checkpoint
├── evaluate.py                 #   Evaluate accuracy
├── benchmark_zero_forgetting.py #  Language-retention benchmark vs. frozen base
├── infer.py                    #   Interactive REPL / batch inference
└── infer_checkpoint.py         #   Inference on raw checkpoints
```

## Architecture Details

The implementation targets **SmolLM2-360M** (LLaMA architecture, 30 layers, hidden dim 1024). The entire base model is frozen — only the ARB injection modules and a LoRA adapter on the LM head are trained.

**ARB placement**: The compute core (stages 1–3) executes once per forward pass. Injection modules are placed at layers 20 and 31 — a mid-network point for early signal propagation and the final layer for direct pre-head injection. Injection occurs after RMSNorm to prevent normalization from diluting the signal.

**LoRA head adapter**: Rank 32, active only at answer positions (detected via the `=` token). This reshapes the output distribution for arithmetic without affecting non-arithmetic generation.

**Parameter budget**:
| Component | Type | Parameters |
|---|---|---|
| Injection modules (×2) | Learned | ~100K |
| LoRA head adapter | Learned | ~1.6M |
| RNS constants + lookup tables | Frozen | ~2K |
| **Total trainable** | | **~1.7M (0.47%)** |
| Base model (SmolLM2-360M) | Frozen | ~360M |

**Training**: Answer-only loss masking — loss is computed only on digit tokens following `=`. The base model's language capabilities are entirely preserved. 250K synthetic examples (pure arithmetic, `A op B = C` format), AdamW with lr 5e-4, batch size 512, up to 200 epochs with early stopping.

## Configuration

All settings live in YAML files mapped to nested dataclasses in `mathllm/config.py`:

| Group | Key Settings (360m config) |
|---|---|
| **RNS** | 6 primes {7..23}, 3 digit slots |
| **ARB** | Layers [20, 31], softmax temp 1000, injection MLP dim 128 |
| **Data** | 250K positive examples, max 3 digits, digit weights [1.5, 3.0, 6.5] |
| **Training** | SmolLM2-360M base, lr 5e-4, batch 512, 200 epochs, LoRA rank 32 |
| **Evaluation** | 1000 samples/config, digits 1-3 |

**Checkpointing**: End-of-epoch checkpoints are always written. `arb_latest.pt` auto-loads on the next training run. Use `--resume PATH` or `--no-resume` to override. `--epochs-to-run` and `--steps-to-run` allow incremental training without resetting optimizer state.

**Model export**: Training auto-exports the final model to `trained_model/`. For early stops, use `scripts/export_checkpoint.py`. Bundles include model weights, config, tokenizer, and base model architecture config.

## Design Decisions

| Decision | Rationale |
|---|---|
| Deterministic extraction | No learned params in stages 1-3. The model writes equations to invoke the ARB — interpretable by construction. No negative examples needed since the ARB only fires on operator tokens. |
| Unconditional execution | All operations run on every token. A router to gate 36 FLOPs would cost more than the computation itself. The injection gate learns relevance via soft weighting. |
| Frozen base model | No catastrophic forgetting. Language capability preserved identically. Only the interface is learned. |
| RNS over binary | Per-prime operations are independent — embarrassingly parallel with no carry propagation. The entire encoding fits in 12 dimensions (<2% of hidden state). |
| High-trust gate init | Gate starts open (σ(3.0) ≈ 0.95) so the model uses the ARB from the start. Training confirms rather than discovers trust. |
| Answer-only loss | Gradients only flow through answer positions. The frozen model's representations for non-arithmetic text are never perturbed. |

## The Research

This implementation validates a general architectural principle: **Don't Learn What You Can Compute.** Every parameter a model spends approximating a deterministic function is a parameter wasted. If a function is exactly computable via tensor operations, embed it as a frozen residual block and let the model learn only the interface.

The pattern is general:
1. Choose an encoding where the target operation becomes a standard tensor operation
2. Implement encoding, computation, and decoding as frozen weights
3. Wrap in deterministic extraction and learned gated injection
4. Insert as a residual block at appropriate layer positions

This applies to any bounded deterministic function expressible as tensor operations — not just arithmetic. The ARB is the proof of concept: it demonstrates that a frozen computation block, inserted as a residual connection, is discoverable by gradient descent and usable by a pretrained model with minimal learned parameters.

The fine-tuning results predict that **pre-training with ARBs** would yield ceiling arithmetic performance across all operations (the accuracy ordering observed in fine-tuning is a fingerprint of the frozen prior, which pre-training eliminates), emergent equation-writing behavior (the gradient reward for exact answers via the ARB dominates internal approximation), and improved non-arithmetic performance (parameters freed from approximating arithmetic are available for language, reasoning, and knowledge).

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.35
- Datasets >= 2.14

See `pyproject.toml` for the full dependency list.
