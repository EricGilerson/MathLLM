# MathLLM

## Don't Learn What You Can Compute

MathLLM implements **Arithmetic Residual Blocks (ARBs)**: frozen modules that
make bounded, exact integer arithmetic available inside a Transformer forward
pass. An ARB validates an explicit terminal equation, computes its result with
a Residue Number System (RNS), and supplies a learned residual/output interface
with the result. The interface learns how to express that result as model
tokens; the arithmetic core itself is fixed.

The motivating design principle is simple: when a function has a cheap exact
implementation and a safe explicit interface, a learned model can use that
computation rather than approximate it solely in learned weights. ARBs are not
a replacement for general tools, retrieval, or interpreters. They are a
concrete in-forward resource for one bounded computation.

## Main results

On the frozen `HuggingFaceTB/SmolLM2-360M` foundation checkpoint, a learned ARB
interface reaches **35,983 / 36,000 (99.9528%)** exact-match accuracy on
supplied one- to three-digit expressions. The matching raw foundation model
gets **1,642 / 36,000 (4.5611%)** under the identical direct prompt protocol.

| Study | What it measures | Result |
| --- | --- | --- |
| Frozen 360M interface | Greedy exact match on supplied direct equations | ARB 99.9528%; raw foundation 4.5611% |
| Compactness sweep | Same frozen model and direct suite, lower LM-head LoRA rank | Rank 1: 99.8944% with 305,602 learned interface parameters; rank 32: 99.9528% with 1,859,074 |
| Joint 20M training | Three seeds; 75% prose, 15% direct equations, 10% text-prefixed equations | ARB/base direct: 97.7% / 19.0%; held-out contextual templates: 99.0% / 37.0%; nearly identical prose NLL |
| Staged matched-capacity diagnostic | Five seeds; arithmetic plus opaque association recall | ARB leads fact acquisition through 5k--20k mixed updates in all seeds and preserves near-perfect arithmetic |

The staged diagnostic is controlled evidence of reduced task interference in a
specific setting, not a demonstration of general reasoning or language-model
improvement. See `review/results.md` and `review/revision_experiments.md` for
the complete evidence ledger and interpretation boundaries.

## Supported interface

The current ARB accepts a tokenizer-valid, terminal equation of the form:

```text
<non-negative integer> <operator> <non-negative integer> =
```

Supported operations are addition, non-negative subtraction, multiplication,
and exact nonzero division. Examples:

```text
35+23=      -> 58
347 * 291 = -> 100977
84/7=       -> 12
```

The syntax-and-domain validator requires contiguous digit tokens, one supported
operator, and the final `=` at the prompt boundary. Whitespace is accepted when
the tokenizer represents it separately. The ARB-specific injection and LM-head
LoRA bypass malformed expressions, completed equations in earlier context,
negative subtraction, zero or inexact division, and values outside the
configured CRT range. Bypass returns control to the frozen base path; it is not
a textual "cannot compute" response.

The six default RNS primes are `{7, 11, 13, 17, 19, 23}`, with product
3,233,230. This covers results from the evaluated three-digit operands.

## How it works

1. **Validate and extract.** A token-level parser recognizes a supported
   terminal equation and extracts its operands.
2. **Encode.** Each operand is represented by its residues modulo the six
   coprime primes, then mapped to points on unit circles.
3. **Compute.** Addition/subtraction use circle operations; multiplication and
   exact division use frozen per-prime tables. CRT reconstructs the bounded
   integer result.
4. **Inject and emit.** A small learned, position-aware MLP injects the result
   at answer positions, and a gated LM-head LoRA maps it to digit logits.

The frozen 360M experiment injects at decoder layer 20 and after the final
RMSNorm (layer 31). The rank-32 interface has 1,859,074 learned parameters;
the deterministic computation has no learned arithmetic parameters.

## Behavior and runtime

The detector runs on every forward call, while the arithmetic core and learned
arithmetic paths run only after a valid trigger. On 256 WikiText-103 documents
(21,239 scored tokens) and 100 SWE-bench Lite patches, no valid trigger fired;
base and ARB scores were identical on those slices. This is a measured
containment result for those distributions, not a guarantee for all text:
quoted valid expressions such as `2+2=5` intentionally satisfy the syntactic
contract and can activate the ARB.

On an A100, no-trigger overhead was +1.97% to +2.81% across 8-, 32-, and
128-token prompts. The current eager-PyTorch active path is +19.8% slower than
the raw base on the measured direct-equation protocol (416.091 ms vs.
347.359 ms). A minimal in-process CPU calculator relay measured 348.774 ms,
so this prototype does **not** claim an active-path latency advantage. See
`scripts/benchmark_runtime.py` and `review/results.md` for the paired protocol
and its boundaries.

## Installation

Python 3.9+ is required. Create an isolated environment and install the
project, including test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional experiment logging:

```bash
pip install -e ".[dev,logging]"
```

PyTorch selects MPS on supported Apple Silicon machines when configured with
`--device mps`; CUDA machines use `--device cuda`.

## Frozen 360M workflow

```bash
# Generate the direct arithmetic training set.
python scripts/generate_data.py --config configs/360m.yaml

# Train the ARB interface. Checkpoints auto-resume by default.
python scripts/train.py --config configs/360m.yaml

# Export and evaluate a standalone model bundle.
python scripts/export_checkpoint.py --config configs/360m.yaml
python scripts/evaluate.py --model-dir trained_model_360m

# Prompt a trained export.
python scripts/infer.py --model-dir trained_model_360m --prompt "347 * 291 ="

# Re-run guarded-domain and containment checks.
python scripts/evaluate_domain_slices.py --model-dir trained_model_360m --device cuda
python scripts/evaluate_containment_probe.py --model-dir trained_model_360m --device cuda

# Measure paired base/ARB/direct-relay timing.
python scripts/benchmark_runtime.py --model-dir trained_model_360m --device cuda
```

Replace `--device cuda` with `--device mps` on Apple Silicon, or omit it for
the scripts' CPU default. Consult each script's `--help` for output paths and
benchmark sizing options.

## From-scratch and capacity experiments

The `mathllm/pretraining/` package contains small from-scratch decoders,
an arithmetic-aware BPE tokenizer, exact block-mixture generation, joint
baseline/ARB training, and associative-recall probes.

```bash
# Build the substantive prose/direct/contextual mixture once.
python scripts/prepare_toy_pretraining_data.py \
  --config configs/toy_pretrain_full.yaml

# Train matched baseline and ARB variants sequentially on that fixed mixture.
python scripts/run_toy_seed.py \
  --config configs/toy_pretrain_full.yaml --skip-prepare --device auto

# Run the staged matched-capacity interference diagnostic.
python scripts/run_staged_reallocation.py \
  --config configs/staged_reallocation_437k_40k_30k.yaml \
  --prepare --device auto
```

These experiments are designed for controlled comparisons. The 20M runs are
practical on CUDA hardware; the smaller staged model can also be used for MPS
smoke tests. Reproduce all reported seeds before treating a configuration as a
new result.

## Repository map

```text
mathllm/
  arb/             frozen RNS core, parser, encoding, and learned injector
  model/           Transformer integration and gated LM-head LoRA
  data/            synthetic direct-arithmetic generation and datasets
  training/        interface training, loss masking, checkpointing
  evaluation/      exact-match and containment utilities
  pretraining/     scratch models, BPE tokenizer, mixtures, and capacity probes
configs/           frozen-model, rank-ablation, pretraining, and staged-study YAMLs
scripts/           training, export, evaluation, containment, and runtime CLIs
review/            internal result ledgers, runbooks, and revision planning
iclr2027/          anonymous ICLR manuscript source and build script
```

## Scope

ARB currently targets supplied, single-step integer equations. It does not yet
evaluate learned equation writing, compound-expression parsing, general tool
selection, or downstream reasoning transfer. The strongest present claim is
that exact in-forward arithmetic is practical under a guarded interface,
compatible with joint training, and associated with reduced interference in a
controlled matched-capacity study.

For the anonymous research repository and paper artifacts, see:

```text
https://anonymous.4open.science/r/MathLLM-FD3A
```
