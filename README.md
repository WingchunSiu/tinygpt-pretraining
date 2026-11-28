# TinyGPT Pretraining

A compact PyTorch codebase for training a GPT-2-style language model from scratch. Built for the TAC 450 capstone project, with character-level Tiny Shakespeare experiments, tiktoken-based OpenWebText training, single-GPU training, and Distributed Data Parallel (DDP).

No pretrained model weights are used. Attention is implemented in PyTorch and dispatches to optimized backend kernels through `torch.nn.functional.scaled_dot_product_attention`.

The original assignment is available in [docs/capstone-project-pretraining-llm.pdf](docs/capstone-project-pretraining-llm.pdf).

## Features

- Decoder-only Transformer with causal multi-head attention, LayerNorm, residual connections, and MLP blocks
- Fixed 512-token context length
- Learned absolute positional embeddings by default, with optional Rotary Position Embeddings (RoPE)
- Character-level tokenizer for Tiny Shakespeare
- GPT-2 `tiktoken` encoding with a 50,257-token vocabulary for OpenWebText
- Non-overlapping next-token prediction sequences with train/validation/test splits
- Configurable model size, batch size, precision, and training duration
- Training loss, validation loss, perplexity, and total tokens/second logging
- Single-node, multi-GPU DDP with per-GPU batch-size reporting
- Checkpointing, text generation, test evaluation, and metric plots

## Repository layout

```text
.
├── model.py
├── train.py
├── sample.py
├── evaluate.py
├── plot_metrics.py
├── requirements.txt
└── data/
    ├── dataset.py
    ├── tokenization.py
    ├── prepare_tiny_shakespeare.py
    └── prepare_openwebtext.py
```

Generated `.bin` datasets, `.pt` checkpoints, and files under `runs/` are excluded from Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch 2.4 or newer is required. On a GPU cluster, install or load the PyTorch build appropriate for the cluster's CUDA version if the default package is not suitable.

## Prepare data

### Tiny Shakespeare

```bash
python data/prepare_tiny_shakespeare.py \
  --output-dir data/tiny_shakespeare
```

This downloads Tiny Shakespeare, constructs a character vocabulary, and writes `train.bin`, `val.bin`, `test.bin`, and `meta.json`.

To use an existing local copy:

```bash
python data/prepare_tiny_shakespeare.py \
  --input path/to/input.txt \
  --output-dir data/tiny_shakespeare
```

### OpenWebText

```bash
python data/prepare_openwebtext.py \
  --subset 10k \
  --output-dir data/openwebtext-10k

python data/prepare_openwebtext.py \
  --subset 100k \
  --output-dir data/openwebtext-100k
```

The script streams `Skylion007/openwebtext` from Hugging Face, shuffles it with a fixed seed, tokenizes each document with GPT-2 `tiktoken`, appends an end-of-text token, and creates 98%/1%/1% train/validation/test document splits. Tokens are stored as `uint16`.

For a small preprocessing check:

```bash
python data/prepare_openwebtext.py \
  --num-documents 100 \
  --shuffle-buffer 10 \
  --output-dir data/openwebtext-smoke
```

Large subsets are best prepared on the training cluster. Streaming requires network access to Hugging Face.

## Train

Start with the default small model on Tiny Shakespeare:

```bash
python train.py \
  --data-dir data/tiny_shakespeare \
  --out-dir runs/shakespeare-small \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256
```

Train a small model on OpenWebText-10k:

```bash
python train.py \
  --data-dir data/openwebtext-10k \
  --out-dir runs/owt10k-small \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256
```

Scale toward GPT-2 Small:

```bash
python train.py \
  --data-dir data/openwebtext-10k \
  --out-dir runs/owt10k-gpt2-small \
  --epochs 3 \
  --batch-size 4 \
  --n-layer 12 \
  --n-head 12 \
  --n-embd 768
```

Adjust the per-GPU batch size to fit the available A40 or L40S memory. Mixed precision defaults to BF16 on CUDA and is disabled on CPU. Use `--dtype float32` or `--dtype float16` to override it.

For a short throughput baseline or smoke run:

```bash
python train.py \
  --data-dir data/openwebtext-10k \
  --out-dir runs/short-baseline \
  --max-iters 300 \
  --batch-size 4 \
  --n-layer 12 \
  --n-head 12 \
  --n-embd 768
```

Run `python train.py --help` for all optimizer, logging, data-loading, and model options.

### Optional RoPE comparison

Keep the core baseline unchanged, then run a separate experiment with the same settings plus `--use-rope`:

```bash
python train.py \
  --data-dir data/tiny_shakespeare \
  --out-dir runs/shakespeare-small-rope \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256 \
  --use-rope
```

RoPE removes the learned positional embedding table and rotates the query and key vectors in each attention head. The checkpoint stores this model setting, so `evaluate.py` and `sample.py` automatically use the correct mode. Compare validation loss, perplexity, and throughput against the baseline only after a full training run; short smoke runs are not meaningful architecture benchmarks.

## Two-GPU DDP

Run inside an allocation with two GPUs on one node:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data-dir data/openwebtext-100k \
  --out-dir runs/owt100k-ddp-2gpu \
  --epochs 3 \
  --batch-size 4 \
  --n-layer 12 \
  --n-head 12 \
  --n-embd 768
```

`--batch-size` is the batch size per GPU. Throughput is measured as total tokens per second across all processes.

For a fair one-GPU versus two-GPU comparison, use the same model, GPU type, precision, and per-GPU batch size. A short one-GPU run can use `--max-iters 300`.

For `N = 2` GPUs, compute scaling efficiency as:

```text
scaling efficiency = N-GPU total throughput / (N × 1-GPU throughput) × 100%
```

## Logs and checkpoints

Each output directory contains:

- `metrics.jsonl`: training and validation records
- `last.pt`: model weights, optimizer state, model configuration, tokenizer metadata, and training arguments
- `best.pt`: checkpoint with the lowest finite validation loss seen during this run (ties retain the earlier checkpoint)

Both checkpoints include their validation loss and the best validation loss seen so far. `last.pt` is updated after every validation; `best.pt` is updated only on improvement, including validation at the end of a partial epoch. Only rank 0 writes checkpoints in DDP. Choose checkpoints using validation loss, not test loss. These changes apply to new runs; an earlier best epoch cannot be recovered from an existing `last.pt`. The trainer does not currently support resuming interrupted runs.

Training loss is recorded every `--log-interval 200` steps and at epoch boundaries. Validation loss and perplexity are recorded after every epoch and after a final partial epoch. By default, validation uses at most 50 batches; change this with `--val-batches`.

## CARC: larger OpenWebText-10k run

`jobs/owt10k-gpt2-small.sbatch` requests one A40, four CPU cores, 32 GB host memory, and four hours on Discovery. It trains a 12-layer, 12-head, 768-dimensional model for three epochs, with batch size 4, BF16, seed 1337, learning rate `6e-4`, logging every 100 steps, and the complete validation split each epoch. It then evaluates both `best.pt` and `last.pt` on the full test split, generates text from `best.pt`, and plots the metrics. Use a fresh output directory for every job.

Before submission:

1. Copy this repository's updated files to your CARC checkout.
2. Reuse your partner's prepared dataset and CUDA-enabled Python environment. Do not copy a Mac virtualenv to CARC.
3. If their environment requires modules, add the same `module load` commands above the activation lines in the batch file. For Conda, replace the `CARC_VENV` check and `source` line with the appropriate Conda shell setup and activation commands.
4. Log in to Discovery and set the following values. Replace the example paths and account; `myaccount` shows available project accounts.

```bash
cd /absolute/path/to/tinygpt-pretraining
export CARC_ACCOUNT='YOUR_PROJECT_ID'
export CARC_VENV='/absolute/path/to/existing/carc-venv'
export DATA_DIR='/absolute/path/to/prepared/openwebtext-10k'
export RUN_BATCH=4
```

Submit a 300-step pilot from the repository root:

```bash
sbatch --account="$CARC_ACCOUNT" --time=00:30:00 \
  --export=ALL,RUN_DIR=runs/owt10k-gpt2-pilot,RUN_MAX_ITERS=300 \
  jobs/owt10k-gpt2-small.sbatch
```

The pilot also exercises evaluation, generation, and plotting. It is not a final training result. Check its Slurm log for CUDA availability, finite loss, and steady throughput. If batch size 4 runs out of GPU memory, set `RUN_BATCH=2` and submit another pilot with a different `RUN_DIR`.

Use pilot throughput to estimate the full run: approximately 33.1 million training tokens divided by tokens/second, plus validation, checkpoint, and evaluation overhead. Adjust the batch file's four-hour time limit if needed, staying within the assignment's 24-hour limit. Submit the full run after the pilot succeeds:

```bash
unset RUN_MAX_ITERS
sbatch --account="$CARC_ACCOUNT" \
  --export=ALL,RUN_DIR=runs/owt10k-gpt2-small \
  jobs/owt10k-gpt2-small.sbatch
```

These batch submissions request their own GPUs; a separate interactive GPU allocation is unnecessary. Do not run training on the login node. Monitor from the login node, replacing `JOB_ID` with the submitted job number:

```bash
squeue -u "$USER"
tail -f slurm-owt10k-gpt2-JOB_ID.out
```

The output directory contains `best.pt`, `last.pt`, `metrics.jsonl`, `test-results-best.json`, `test-results-last.json`, `sample-best.txt`, and `plots/`. Keep the Slurm log as well. Compare best and final checkpoints separately, and label their selection method when comparing against the earlier small-model final checkpoint.

## Generate text

Tiny Shakespeare example:

```bash
python sample.py \
  --checkpoint runs/shakespeare-small/last.pt \
  --prompt "First Citizen:" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 200
```

OpenWebText example:

```bash
python sample.py \
  --checkpoint runs/owt10k-small/last.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```

The tokenizer is selected from the checkpoint metadata.

## Evaluate and plot

Evaluate the full test split and compute perplexity:

```bash
python evaluate.py \
  --checkpoint runs/shakespeare-small/last.pt \
  --data-dir data/tiny_shakespeare \
  --split test \
  --batch-size 16 \
  --output runs/shakespeare-small/test-results.json
```

Use `--max-batches` for a quicker partial evaluation.

Generate loss and throughput plots:

```bash
python plot_metrics.py \
  --metrics runs/shakespeare-small/metrics.jsonl \
  --output-dir runs/shakespeare-small/plots
```

This writes `loss.png` and `throughput.png`.

Completed core-run configurations, metrics, DDP scaling results, and report notes are recorded in [docs/core-runs.md](docs/core-runs.md).

## Potential next steps

### QK RMSNorm

Apply RMSNorm independently to the query and key tensors in each attention head before scaled dot-product attention. Compare training stability, validation loss, perplexity, and throughput against the unchanged baseline. An optional higher-learning-rate run can test whether QK normalization improves robustness to gradient spikes.

Run the baseline, RoPE, and QK RMSNorm experiments separately so that each result isolates one architectural change.
