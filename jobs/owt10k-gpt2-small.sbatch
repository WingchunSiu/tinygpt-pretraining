#!/bin/bash
#SBATCH --job-name=owt10k-gpt2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Submit from the repository root. Supply your course account via sbatch --account.
cd "${SLURM_SUBMIT_DIR:?Submit this file using sbatch from the repository root}"

# Match the modules and environment created by carc-setup.sbatch.
module load "${CARC_GCC_MODULE:-gcc/13.3.0}" "${CARC_PYTHON_MODULE:-python/3.11.9}"
CARC_VENV="${CARC_VENV:-$PWD/.venv-carc}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PWD/.cache/tiktoken}"
if [[ ! -f "$CARC_VENV/bin/activate" ]]; then
    echo "Environment missing. Submit jobs/carc-setup.sbatch first." >&2
    exit 1
fi
source "$CARC_VENV/bin/activate"

RUN_DIR="${RUN_DIR:-runs/owt10k-gpt2-small}"
RUN_LR="${RUN_LR:-6e-4}"
RUN_BATCH="${RUN_BATCH:-4}"
DATA_DIR="${DATA_DIR:-data/openwebtext-10k}"
# Set RUN_MAX_ITERS=300 for a pilot; unset it for the full three-epoch run.
set --
if [[ -n "${RUN_MAX_ITERS:-}" ]]; then
    set -- --max-iters "$RUN_MAX_ITERS"
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "Choose a fresh RUN_DIR: $RUN_DIR already exists." >&2
    exit 1
fi
for split_file in meta.json train.bin val.bin test.bin; do
    if [[ ! -s "$DATA_DIR/$split_file" ]]; then
        echo "Missing or empty dataset file: $DATA_DIR/$split_file" >&2
        exit 1
    fi
done

nvidia-smi
python -u - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is unavailable; check the GPU allocation and PyTorch environment"
assert torch.cuda.is_bf16_supported(), "This job requires BF16 support"
# Record the software and source revision alongside Slurm's hardware information.
print(json.dumps({"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0)}, indent=2), flush=True)
PY
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    git rev-parse HEAD
    git status --short
fi

python -u train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$RUN_DIR" \
    --epochs 3 \
    --batch-size "$RUN_BATCH" \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --learning-rate "$RUN_LR" \
    --dtype bfloat16 \
    --seed 1337 \
    --log-interval 100 \
    --val-batches 1000000 \
    "$@"

# Select by validation loss, then evaluate the full test split. Preserve the
# final checkpoint's result too, so overfitting is visible in the report.
for checkpoint in best last; do
    python -u evaluate.py \
        --checkpoint "$RUN_DIR/$checkpoint.pt" \
        --data-dir "$DATA_DIR" \
        --split test \
        --batch-size "$RUN_BATCH" \
        --output "$RUN_DIR/test-results-$checkpoint.json"
done

python -u sample.py \
    --checkpoint "$RUN_DIR/best.pt" \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 200 \
    --seed 1337 \
    > "$RUN_DIR/sample-best.txt"

python -u plot_metrics.py \
    --metrics "$RUN_DIR/metrics.jsonl" \
    --output-dir "$RUN_DIR/plots"

echo "Completed run: $RUN_DIR"
