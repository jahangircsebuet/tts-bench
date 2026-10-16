#!/bin/bash
# Gemini generation driver. Requires GEMINI_API_KEY to already be exported
# in the environment (see ../../README.md, "Environment").
set -uo pipefail
cd "$(dirname "$0")"
LOGS=../logs

run() {
  echo "=== $(date '+%F %T') START: $* ===" | tee -a "$LOGS/run_gemini.log"
  python3 "$@" >> "$LOGS/run_gemini.log" 2>&1
  code=$?
  echo "=== $(date '+%F %T') END ($code): $* ===" | tee -a "$LOGS/run_gemini.log"
  return $code
}

echo "########## gemini-3.5-flash-lite: unsafe ##########" | tee -a "$LOGS/run_gemini.log"
run gen_gemini.py --dataset unsafe --concurrency 12

echo "########## gemini-3.5-flash-lite: benign ##########" | tee -a "$LOGS/run_gemini.log"
run gen_gemini.py --dataset benign --concurrency 12

echo "GEMINI_GEN_DONE $(date '+%F %T')" | tee -a "$LOGS/run_gemini.log"
# NOTE: judging runs separately (run_gemini_judge.sh) after three-model-run's
# GPU job finishes.
