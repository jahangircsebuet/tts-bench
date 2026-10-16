#!/bin/bash
# Judging for gemini_flash_lite.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.
LOGS=../logs

run() {
  echo "=== $(date '+%F %T') START: $* ===" | tee -a "$LOGS/run_closed.log"
  python3 "$@" >> "$LOGS/run_closed.log" 2>&1
  code=$?
  echo "=== $(date '+%F %T') END ($code): $* ===" | tee -a "$LOGS/run_closed.log"
  return $code
}

for ds in unsafe benign; do
  for j in qwen3_14b aya_8b; do
    run judge_closed.py --model gemini_flash_lite --dataset $ds --judge $j
  done
done

run analyze_closed.py
echo "GEMINI_JUDGE_DONE $(date '+%F %T')" | tee -a "$LOGS/run_closed.log"
