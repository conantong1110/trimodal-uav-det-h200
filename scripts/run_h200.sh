#!/usr/bin/env bash
# Train and then evaluate TriModalDet on a single NVIDIA H200.
#
# Usage:
#   bash scripts/run_h200.sh
#
# Settings are environment variables, for example:
#   DATA=/path/to/data BACKBONE=mit_b1 bash scripts/run_h200.sh
#
#   DATA              data folder containing images/ and labels/   (default: data)
#   BACKBONE          mit_b0 .. mit_b4                             (default: mit_b1)
#   EPOCHS            training epochs                              (default: 15)
#   BATCH_SIZE        batch size                                   (default: 16)
#   LR                learning rate; the paper uses 0.02 at batch 16 and
#                     scales it linearly with the batch size       (default: 0.02)
#   WORKERS           DataLoader worker processes                  (default: 8)
#   GPU               GPU index; unset keeps the scheduler's choice
#   WATCHDOG_MINUTES  kill the run after this long without a log   (default: 40)
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
BACKBONE=${BACKBONE:-mit_b1}
EPOCHS=${EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-0.02}
WORKERS=${WORKERS:-8}
export TRIMODAL_WATCHDOG_MINUTES=${WATCHDOG_MINUTES:-40}
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
fi

mkdir -p checkpoints logs
RUN=${BACKBONE}_$(date +%Y%m%d_%H%M%S)
MODEL=checkpoints/${RUN}.pth
LOG=logs/${RUN}.log

echo "Run: ${RUN}  (log: ${LOG})" | tee "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

# pipefail makes a failed or watchdog-killed training stop the script here.
python -u scripts/train.py --data "$DATA" --backbone "$BACKBONE" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --lr "$LR" --workers "$WORKERS" --model "$MODEL" 2>&1 | tee -a "$LOG"

python -u scripts/test.py --data "$DATA" --backbone "$BACKBONE" --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" --model "$MODEL" --results-dir "test_results/${RUN}" 2>&1 | tee -a "$LOG"
