#!/bin/bash
#SBATCH --job-name=llm_scoring
#SBATCH --output=scoring_%j.out
#SBATCH --error=scoring_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn05,cn06

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

module purge
module load python/3.12.9
module load cuda/12.8

cd "$HOME/LLM-Routing-Project"

echo "Current directory: $(pwd)"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR="llama-32-1b"          # llama-32-1b | llama-32-3b | mistral-7b | mistral-nemo-12b
CHUNK="0"                        # 0 | 2000 | 4000 | 6000 | 8000
SCORER="bart"                  # armorm | bart
# MODEL_NAME=""                  # optional: uncomment to override the scorer's default model

DATA_PATH="outputs/${MODEL_DIR}/responses/${MODEL_DIR}_${CHUNK}.jsonl"
OUT_PATH="outputs/${MODEL_DIR}/scores/${SCORER}/${MODEL_DIR}_${CHUNK}_scored.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

echo "----------------------------------------"
echo "Scorer     : $SCORER"
echo "Input file : $DATA_PATH"
echo "Output file: $OUT_PATH"
echo "----------------------------------------"

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Input file not found: $DATA_PATH"
    exit 1
fi

echo "Starting scoring..."

EXTRA_ARGS=""
if [ -n "${MODEL_NAME:-}" ]; then
    EXTRA_ARGS="--model-name $MODEL_NAME"
fi

srun python main.py score \
    --data-path "$DATA_PATH" \
    --out-path  "$OUT_PATH" \
    --scorer    "$SCORER" \
    $EXTRA_ARGS

echo "----------------------------------------"
echo "Scoring completed at: $(date)"