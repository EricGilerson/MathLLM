# MathLLM

**Arithmetic Residual Blocks: Exact Computation Native to the Transformer Forward Pass**

MathLLM augments a frozen GPT-2 with *Arithmetic Residual Blocks* (ARBs) — small, surgically inserted modules that perform exact integer arithmetic using the Residue Number System (RNS) and unit-circle encoding. No tool calls, no CUDA stream breaks, no external calculators. Arithmetic happens inside the forward pass as standard tensor operations.

## Key Idea

Integers are encoded as points on unit circles via RNS. Addition becomes complex multiplication (rotation), subtraction becomes conjugation + rotation, and multiplication/exponentiation use small frozen lookup tables. Nine coprime primes `{7, 11, 13, 17, 19, 23, 29, 31, 37}` cover integers up to ~7.4 x 10^9, requiring only 18 dimensions (< 1% of GPT-2's hidden state). The float32 precision margin is 8 orders of magnitude — the encoding is exact, not approximate.

## Architecture

Each ARB is a 4-stage pipeline inserted after specific transformer layers (default: 4, 8, 10):

```
Hidden State h
    |
[Stage 1: Extract]  -- Learned linear projections → two digit vectors (STE rounding)
    |
[Stage 2: Encode]   -- Frozen RNS circle encoding (cos/sin per prime)
    |
[Stage 3: Compute]  -- Frozen parallel arithmetic: +, −, ×, ^ (all four, every token)
    |
[Stage 4: Inject]   -- Learned projection back to hidden dim (zero-initialized, residual)
    |
Hidden State h' = h + delta
```

- **Stages 1 & 4** are learned (~47K parameters per ARB)
- **Stages 2 & 3** are frozen (precomputed constants and lookup tables)
- **Base GPT-2** is completely frozen (124M parameters untouched)
- **Total trainable overhead**: ~141K parameters (0.11% of base model)

The zero-initialization of Stage 4 means the model starts as vanilla GPT-2 and gradually learns when and how to use arithmetic.

## Project Structure

```
mathllm/
├── arb/                    # Arithmetic Residual Block implementation
│   ├── arb_module.py       #   Main ARB orchestrator (4-stage pipeline)
│   ├── stage1_extract.py   #   Learned operand extraction with STE
│   ├── stage2_encode.py    #   Frozen RNS circle encoding
│   ├── stage3_compute.py   #   Frozen arithmetic (add, sub, mul, exp)
│   ├── stage4_inject.py    #   Learned result injection (zero-init)
│   ├── constants.py        #   Precomputed RNS constants and lookup tables
│   └── ste.py              #   Straight-Through Estimator for gradient flow
├── data/                   # Data generation pipeline
│   ├── generator.py        #   Synthetic example generation (pos/neg/edge)
│   ├── dataset.py          #   PyTorch Dataset with tokenization & augmentation
│   ├── templates.py        #   1100+ natural-language arithmetic templates
│   └── negative_examples.py#   Non-arithmetic examples with numbers
├── model/                  # Model integration
│   ├── gpt2_arb.py         #   GPT2WithARB (frozen base + ARB insertion)
│   └── utils.py            #   Model utilities
├── training/
│   └── trainer.py          #   Training loop (AdamW, warmup, early stopping)
├── evaluation/
│   └── evaluator.py        #   Exact-match accuracy by digit count and operation
└── config.py               # Nested dataclass config with YAML loading

configs/
├── default.yaml            # Full training config (110K examples, 10 epochs)
└── debug.yaml              # Small config for fast iteration

scripts/
├── generate_data.py        # Generate synthetic arithmetic dataset
├── train.py                # Run training
└── evaluate.py             # Evaluate model accuracy
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Generate training data
python scripts/generate_data.py --config configs/default.yaml

# Train
python scripts/train.py --config configs/default.yaml

# Run a small incremental chunk; training auto-resumes from checkpoints/latest
python scripts/train.py --config configs/default.yaml --epochs-to-run 2

# Or budget by optimizer steps instead of epochs
python scripts/train.py --config configs/default.yaml --steps-to-run 1000

# Training also exports a standalone final model bundle
# to trained_model/ by default, or a custom path via --export-dir

# Evaluate
python scripts/evaluate.py --config configs/default.yaml

# Evaluate an exported final model bundle directly
python scripts/evaluate.py --model-dir trained_model/

# Fast iteration (smaller dataset, fewer epochs)
python scripts/train.py --config configs/debug.yaml
```

## Training Data

The data pipeline generates three categories of examples:

| Category | Default Count | Purpose |
|---|---|---|
| **Positive** | 50,000 | Arithmetic in diverse formats (equations, natural language, LaTeX, code, word problems) |
| **Negative** | 50,000 | Numbers in non-arithmetic contexts (flights, dates, page numbers, port numbers) |
| **Edge cases** | 10,000 | Zero, identity, boundary values, negative results |

Templates span 1100+ formats per operation to prevent extraction-layer overfitting. Operations include single-step (+, -, *, ^, exact /) and multi-step chains (2-3 operations composed).

## How It Works

**RNS Circle Encoding**: An integer `n` is mapped to 9 residues `(n mod 7, n mod 11, ..., n mod 37)`, each encoded as a point on a unit circle: `(cos(2*pi*r/p), sin(2*pi*r/p))`.

**Addition**: Complex multiplication of circle points — `e^(i*a) * e^(i*b) = e^(i*(a+b))` — gives exact modular addition. Cost: 6 FLOPs per prime, 54 total.

**Multiplication**: Soft-decode residues via template inner products + softmax, compute outer product, index into a frozen `p x p` lookup table per prime, re-encode result.

**Exponentiation**: Fermat's Little Theorem reduces `a^b mod p` to `a^(b mod (p-1)) mod p`, then a single table lookup per prime.

**Gradient Flow**: Stages 2 & 3 are frozen but differentiable. The Straight-Through Estimator in Stage 1 allows gradients to flow through rounding. Soft one-hot decoding in Stage 3 maintains gradient paths through multiplication/exponentiation.

## Configuration

All settings are controlled via YAML configs with dataclass defaults:

| Group | Key Settings |
|---|---|
| **RNS** | 9 primes, 10 digit slots |
| **ARB** | Layer positions `[4, 8, 10]`, softmax temp 1000, dropout 0.1 |
| **Data** | 50K/50K/10K split, max 10 digits, max value 10^9 |
| **Training** | GPT-2 base, lr 3e-4, batch 32, 10 epochs, warmup 500 steps, early stopping (patience 3) |
| **Evaluation** | 200 samples/config, digits 1-10 |

Checkpointing and resume behavior:

- End-of-epoch checkpoints are always written.
- Optional step checkpoints are controlled by `training.checkpoint_every_steps`.
- `checkpoints/arb_latest.pt` is refreshed on every checkpoint and auto-loaded by default on the next training run.
- Use `--resume PATH` to force a specific checkpoint, or `--no-resume` to start fresh.
- Use `--epochs-to-run N` or `--steps-to-run N` to train in small chunks without resetting optimizer or LR progress.

Final model export:

- Every training run exports the in-memory final model to `training.final_model_dir`.
- The export bundle includes `model_state.pt`, `config.yaml`, tokenizer files, and the saved GPT-2 architecture config in `base_model_config/`.
- Override the destination with `--export-dir PATH`.
- Load the bundle with `python scripts/evaluate.py --model-dir trained_model/`.

## Design Decisions

| Decision | Rationale |
|---|---|
| RNS over binary | Embarrassingly parallel per-prime ops, no carry propagation |
| Circle encoding | Trig periodicity gives free modular reduction; enables differentiable soft decoding |
| Unconditional execution | All 4 ops run on every token; injection layer learns relevance |
| Zero-init injection | ARB starts as identity — no risk of corrupting pretrained representations |
| Frozen base model | No catastrophic forgetting; language capability preserved |
| 50/50 positive/negative data | Prevents hallucinating arithmetic into non-math text |
| 1100+ templates | Format diversity prevents extraction overfitting |

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.35
- Datasets >= 2.14

See `pyproject.toml` for the full dependency list.

## Further Reading

See [`idea.md`](idea.md) for the full technical writeup, including mathematical derivations, precision analysis, and extensions to transcendental functions.
