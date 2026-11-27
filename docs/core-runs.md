# Core Training Runs

Date started: 2026-11-27

This is the record for the completed small-model core runs. It preserves the evidence needed for the final report and presentation. All reported values come from completed run artifacts, not from smoke tests.

## Assignment requirements

The assignment in `docs/capstone-project-pretraining-llm.pdf` gives this progression:

1. Train a small model on Tiny Shakespeare for early experiments.
2. Switch to GPT-2 `tiktoken` encoding with a 50,257-token vocabulary and train on OpenWebText-10k, starting with a small model to experiment with hyperparameters.
3. Train for 3–5 epochs, logging training loss every few hundred steps, validation loss at least once per epoch, and throughput in tokens per second.
4. Scale OpenWebText-10k toward GPT-2 Small on one GPU.
5. Train OpenWebText-100k for 3–5 epochs with two-GPU DDP and compare one-GPU and two-GPU throughput and scaling efficiency.
6. Report loss and throughput plots, test perplexity, generated samples, findings, limitations, and future work.

Tiny Shakespeare and the OpenWebText-10k small run are therefore core experiments, not merely smoke tests. Their purpose is to validate the training setup and hyperparameters at a lower cost before longer, larger runs. The earlier 137,024-parameter, three-step `smoke-tiny` run only verified the software pipeline and must not be reported as a final model result.

## Current run registry

| Run | Slurm job | Final status | Purpose |
| --- | --- | --- | --- |
| Tiny Shakespeare small | 14965002 | COMPLETED | Core small-model training |
| OpenWebText-10k preprocessing | 14965004 | COMPLETED | Prepared 10,000 documents |
| OpenWebText-10k small | 14965005 | COMPLETED | Core small-model training |
| OpenWebText-100k preprocessing | 14965753 | COMPLETED | Prepared 100,000 documents |
| OpenWebText-100k two-GPU DDP | 14965755 | COMPLETED | Same small architecture, three epochs |
| OpenWebText-100k one-GPU baseline | 14965756 | COMPLETED | Matching 300-step throughput baseline |

An earlier preprocessing job, 14965001, failed because the current Hugging Face stack rejected the unqualified `openwebtext` repository ID. The default was corrected to `Skylion007/openwebtext`; its dependent training job was cancelled before using any GPU resources.

## Shared protocol

- Architecture: decoder-only GPT with learned absolute positional embeddings; RoPE is not used in these baseline runs.
- Context length: 512 tokens.
- Layers: 4.
- Attention heads: 4.
- Embedding dimension: 256.
- Epochs: 3.
- Batch size: 16 sequences per GPU.
- GPUs: 1 requested NVIDIA A40 per training run.
- Seed: 1337.
- Learning rate: 6e-4.
- Weight decay: 0.1.
- AdamW betas: 0.9 and 0.95.
- Gradient clipping: 1.0.
- Dropout: 0.0.
- Precision: BF16 on CUDA.
- Validation: after every epoch and after a final partial epoch, using at most 50 batches.
- Logging: training loss and total tokens/second every 200 steps and at epoch boundaries.

The parameter count, exact steps per epoch, actual GPU model, and elapsed time must be taken from each run's Slurm log.

## Datasets

### Tiny Shakespeare

- Tokenizer: character-level.
- Vocabulary size: 65.
- Training tokens: 1,003,854.
- Validation tokens: 55,770.
- Test tokens: 55,770.
- Metadata: `data/tiny_shakespeare/meta.json`.

### OpenWebText-10k

- Source: `Skylion007/openwebtext`, streamed from Hugging Face.
- Tokenizer: GPT-2 `tiktoken` encoding.
- Vocabulary size: 50,257.
- Requested documents: 10,000, shuffled with seed 1337.
- Split: 98% training, 1% validation, 1% test by document.
- Documents: 9,800 training, 100 validation, 100 test.
- Tokens: 11,032,491 training, 110,833 validation, 111,218 test.
- Metadata: `data/openwebtext-10k/meta.json`.

## Reproduction commands

### Tiny Shakespeare small

```bash
python train.py \
  --data-dir data/tiny_shakespeare \
  --out-dir runs/shakespeare-small \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256

python evaluate.py \
  --checkpoint runs/shakespeare-small/last.pt \
  --data-dir data/tiny_shakespeare \
  --split test \
  --batch-size 16 \
  --output runs/shakespeare-small/test-results.json

python sample.py \
  --checkpoint runs/shakespeare-small/last.pt \
  --prompt "First Citizen:" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 200

python plot_metrics.py \
  --metrics runs/shakespeare-small/metrics.jsonl \
  --output-dir runs/shakespeare-small/plots
```

### OpenWebText-10k small

```bash
python data/prepare_openwebtext.py \
  --subset 10k \
  --output-dir data/openwebtext-10k

python train.py \
  --data-dir data/openwebtext-10k \
  --out-dir runs/owt10k-small \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256

python evaluate.py \
  --checkpoint runs/owt10k-small/last.pt \
  --data-dir data/openwebtext-10k \
  --split test \
  --batch-size 16 \
  --output runs/owt10k-small/test-results.json

python sample.py \
  --checkpoint runs/owt10k-small/last.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 200

python plot_metrics.py \
  --metrics runs/owt10k-small/metrics.jsonl \
  --output-dir runs/owt10k-small/plots
```

## Completed results

| Metric | Tiny Shakespeare small | OpenWebText-10k small |
| --- | --- | --- |
| Final job state | COMPLETED | COMPLETED |
| Parameters | 3,307,264 | 16,156,416 |
| Training tokens processed | 3,010,560 | 33,096,192 |
| Steps per epoch / total steps | 123 / 369 | 1,347 / 4,041 |
| Slurm elapsed time | 23 seconds | 3 minutes 7 seconds |
| Final logged training loss | 2.3652 | 5.1279 |
| Final validation loss | 2.3648 | 5.1606 |
| Final validation perplexity | 10.6424 | 174.2665 |
| Test loss | 2.3842 | 5.1888 |
| Test perplexity | 10.8505 | 179.2451 |
| Test sequences | 108 | 217 |
| Steady-state median throughput | 1,038,150 tokens/s | 196,095 tokens/s |

Slurm elapsed time includes training, full test evaluation, generation, and plotting. Final logged training loss is the mean over the last logging window, not a single-batch loss. Throughput is the median of training records after excluding the first record, which includes startup overhead. Token counts exclude incomplete trailing blocks and exactly match the sequences processed during three epochs. Do not compare the two datasets' perplexities directly: their tokenizers and vocabularies differ.

### Epoch-level validation history

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| Tiny Shakespeare validation loss | 2.4908 | 2.4358 | 2.3648 |
| Tiny Shakespeare validation perplexity | 12.0711 | 11.4251 | 10.6424 |
| OpenWebText-10k validation loss | 5.6346 | 5.3239 | 5.1606 |
| OpenWebText-10k validation perplexity | 279.9512 | 205.1867 | 174.2665 |

Validation loss decreased throughout both three-epoch runs, so these small models did not yet show validation-loss overfitting. This does not rule out overfitting when scaling to a larger model, which the assignment specifically asks us to investigate on OpenWebText-10k.

The Tiny Shakespeare sample learned character patterns, line breaks, and speaker-like formatting, but its words and meaning remained largely incoherent. The OpenWebText sample produced recognizable English sentences and a clear topical pattern, but was generic and repetitive. These qualitative observations should accompany the samples in the report rather than presenting either small model as a coherent final generator.

All four plots were opened and checked after generation. The loss curves agree with the JSONL metrics, and the throughput plots show the expected lower startup measurement followed by stable throughput.

## OpenWebText-100k DDP experiment

The 100k run keeps the 16.2M-parameter small architecture unchanged: 4 layers, 4 attention heads, embedding dimension 256, context length 512, and three epochs. It uses two A40 GPUs on one node with DDP and a per-GPU batch size of 16. The effective batch size is therefore 32 sequences per step, compared with 16 for the earlier 10k run. The data-scale comparison thus measures the practical combined effect of more data, a larger global batch, and a different number of optimizer updates; it does not isolate dataset size alone.

The dataset contains 98,000 training, 1,000 validation, and 1,000 test documents, with 110,907,512 training, 1,146,959 validation, and 1,123,453 test tokens. Three epochs processed 332,722,176 training tokens in 20,310 optimizer steps, or 6,770 steps per epoch.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data-dir data/openwebtext-100k \
  --out-dir runs/owt100k-ddp-2gpu \
  --epochs 3 \
  --batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256
```

A separate 300-step, one-A40 run used the same model, precision, seed, and per-GPU batch size. Scaling efficiency is calculated from steady-state median total throughput:

```text
scaling efficiency = two-GPU total throughput / (2 × one-GPU throughput) × 100%
```

### Throughput and scaling results

| Metric | One A40, 300 steps | Two A40s, three epochs |
| --- | --- | --- |
| Slurm job | 14965756 | 14965755 |
| Per-GPU batch size | 16 | 16 |
| Global batch size | 16 | 32 |
| Steady-state median total throughput | 196,762 tokens/s | 336,360 tokens/s |
| Slurm elapsed time | 47 seconds | 17 minutes 4 seconds |

The two-GPU speedup is 1.71× and scaling efficiency is **85.47%**. The short one-GPU baseline has only one steady-state throughput record after startup, so its estimate is less robust than the long DDP run's median. Both jobs used A40 GPUs, but they were scheduled on different nodes.

### Validation and test results

| Metric | OpenWebText-100k DDP |
| --- | --- |
| Parameters | 16,156,416 |
| Final logged training loss | 4.5048 |
| Final validation loss | 4.5275 |
| Final validation perplexity | 92.5261 |
| Test loss | 4.5483 |
| Test perplexity | 94.4726 |
| Test sequences | 2,194 |

Validation loss by epoch was **4.6015 → 4.4802 → 4.5275**, with validation perplexity **99.6320 → 88.2524 → 92.5261**. Epoch 2 had the best validation loss; its increase in epoch 3 is an early overfitting signal. The current checkpointing scheme retains `last.pt`, so the reported test result is for the final epoch-3 checkpoint, not the better epoch-2 validation state.

For a fair data-scale comparison, both the OpenWebText-10k and OpenWebText-100k checkpoints were evaluated on the same OpenWebText-100k test split. The 10k model has not seen those documents. The reverse comparison would risk leakage because documents assigned to the 10k validation/test splits fall inside the larger 100k training split. The original 10k test result remains preserved separately.

| Checkpoint, evaluated on the 100k test split | Test loss | Test perplexity |
| --- | --- | --- |
| OpenWebText-10k small | 5.2532 | 191.1764 |
| OpenWebText-100k two-GPU DDP | 4.5483 | 94.4726 |

The 100k checkpoint reduced common-test loss by 0.7049 and perplexity by 50.58%. This is a substantial practical improvement, although dataset size, global batch size, and optimizer step count differ between the runs. The generated 100k sample remains repetitive despite the improved perplexity, illustrating that lower loss does not guarantee high-quality open-ended generation.

All DDP and one-GPU plots were opened and checked against the JSONL records. Artifacts are stored under:

- `runs/owt100k-ddp-2gpu/`
- `runs/owt100k-baseline-1gpu/`
- `runs/owt10k-small/test-results-owt100k.json`

## Artifacts for report and presentation

Each completed run contains:

- `slurm-<jobid>.out`: hardware, configuration, metrics, generated sample for full runs, and errors if any.
- `metrics.jsonl`: training and validation history.
- `last.pt`: model, optimizer, model configuration, tokenizer metadata, and training arguments.
- `test-results.json`: full test-split loss and perplexity for the full training runs; the short throughput baseline does not require a separate test result.
- `plots/loss.png`: training and validation loss versus step.
- `plots/throughput.png`: throughput versus step.

The report should explain the two tokenizers, model dimensions, parameter counts, token budgets, training/validation curves, test perplexity, sample quality, and throughput. For OpenWebText-10k, explicitly inspect validation loss for overfitting; the assignment warns that larger models are expected to overfit this roughly 10M-token subset.

## Remaining assignment work after these runs

- Scale OpenWebText-10k toward GPT-2 Small on one GPU for 3–5 epochs.
- Optional: RoPE, QK RMSNorm, or four-GPU/two-node experiments, clearly separated from the required baseline results.
