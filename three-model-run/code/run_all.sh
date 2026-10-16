#!/bin/bash
# Full three-model-run pipeline, in the user-specified order:
#   1) unsafe generation, one model at a time (voxtral, phi4mm, qwen2audio)
#   2) benign generation, one model at a time (same order)
#   3) judging (per-model judge pair) for both datasets
#   4) analysis
# GPU 2, vLLM, least-possible gpu_memory_utilization (set per-model in gen_model.py).
# Resumable: every stage skips items/files already written.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.
LOGS=../logs

run() {
  echo "=== $(date '+%F %T') START: $* ===" | tee -a "$LOGS/run_all.log"
  python3 "$@" >> "$LOGS/run_all.log" 2>&1
  code=$?
  echo "=== $(date '+%F %T') END ($code): $* ===" | tee -a "$LOGS/run_all.log"
  return $code
}

MODELS="voxtral phi4mm qwen2audio"

echo "########## PHASE 1: unsafe generation ##########" | tee -a "$LOGS/run_all.log"
for m in $MODELS; do
  run gen_model.py --model $m --dataset unsafe
done

echo "########## PHASE 2: benign generation ##########" | tee -a "$LOGS/run_all.log"
for m in $MODELS; do
  run gen_model.py --model $m --dataset benign
done

echo "########## PHASE 3: judging ##########" | tee -a "$LOGS/run_all.log"
for m in $MODELS; do
  for ds in unsafe benign; do
    for j in qwen3_14b aya_8b gemma3_12b; do
      run judge_model.py --model $m --dataset $ds --judge $j
    done
  done
done

echo "########## PHASE 4: analysis ##########" | tee -a "$LOGS/run_all.log"
run analyze_three_models.py

echo "ALL_DONE $(date '+%F %T')" | tee -a "$LOGS/run_all.log"
