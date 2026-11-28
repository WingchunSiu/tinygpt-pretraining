#!/bin/bash
#SBATCH --job-name=tinygpt-setup
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# CPU-only setup: install the GPU-capable environment and prepare data before
# reserving a GPU. Run from a CARC checkout; no partner's environment is needed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"

module load "${CARC_GCC_MODULE:-gcc/13.3.0}" "${CARC_PYTHON_MODULE:-python/3.11.9}"
CARC_VENV="${CARC_VENV:-$PWD/.venv-carc}"
DATA_DIR="${DATA_DIR:-$PWD/data/openwebtext-10k}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PWD/.cache/tiktoken}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! -f "$CARC_VENV/pyvenv.cfg" ]]; then
    if [[ -e "$CARC_VENV" ]]; then
        echo "Existing path is not a virtualenv: $CARC_VENV. Choose a new CARC_VENV." >&2
        exit 1
    fi
    python -m venv "$CARC_VENV"
fi
source "$CARC_VENV/bin/activate"
python -m pip install --upgrade pip
# Use a fixed CUDA wheel, even though this setup job has no GPU.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > "$CARC_VENV/requirements-lock.txt"

python - <<'PY'
import torch
assert torch.version.cuda is not None, "Installed PyTorch does not include CUDA"
print(f"Installed PyTorch {torch.__version__}, CUDA build {torch.version.cuda}", flush=True)
print("CUDA device availability is checked later by the GPU pilot.", flush=True)
PY

if [[ ! -e "$DATA_DIR" ]]; then
    mkdir -p "$(dirname "$DATA_DIR")"
    # Publish only after preprocessing finishes. On failure, the staging folder
    # remains for inspection; an incomplete dataset is never silently reused.
    staging_dir=$(mktemp -d "${DATA_DIR}.preparing.XXXXXX")
    python -u data/prepare_openwebtext.py \
        --subset 10k \
        --seed 1337 \
        --output-dir "$staging_dir"
    if [[ -e "$DATA_DIR" ]]; then
        echo "Another process created $DATA_DIR; inspect $staging_dir before retrying." >&2
        exit 1
    fi
    mv "$staging_dir" "$DATA_DIR"
else
    echo "Checking existing dataset without overwriting: $DATA_DIR"
fi

python - "$DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "meta.json").open() as file:
    meta = json.load(file)
expected = {"tokenizer": "tiktoken", "encoding": "gpt2", "vocab_size": 50257,
            "block_size": 512, "seed": 1337}
for key, value in expected.items():
    if meta.get(key) != value:
        raise ValueError(f"{root}: expected {key}={value!r}, got {meta.get(key)!r}")
if meta.get("dataset") != "Skylion007/openwebtext":
    raise ValueError(f"Unexpected dataset source in {root}")
if meta.get("num_documents") != {"train": 9800, "val": 100, "test": 100}:
    raise ValueError(f"Expected a complete 10k-document dataset in {root}")
for split in ("train", "val", "test"):
    tokens = meta["num_tokens"][split]
    if tokens < 513 or (root / f"{split}.bin").stat().st_size != 2 * tokens:
        raise ValueError(f"Invalid uint16 token file: {root / (split + '.bin')}")
print(json.dumps(meta, indent=2), flush=True)
PY

echo "Setup complete. Environment: $CARC_VENV; dataset: $DATA_DIR"
