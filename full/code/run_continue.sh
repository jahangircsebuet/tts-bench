#!/bin/bash
# after Piper fix: speech pass for new rows -> judges -> TTS quality -> metrics
cd "$(dirname "$0")"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
python3 F03_run_audio_model.py --model gemma3n  --gpu-memory-utilization 0.4 --conditions speech_lang > ../logs/F03c_gemma3n_piper.log 2>&1
python3 F03_run_audio_model.py --model ultravox --gpu-memory-utilization 0.3 --conditions speech_lang > ../logs/F03c_ultravox_piper.log 2>&1
for j in gemma3_12b qwen3_14b aya_8b; do
  python3 F04_judge_responses.py --judge $j > ../logs/F04c_$j.log 2>&1
done
python3 F05_tts_quality.py > ../logs/F05_tts_quality.log 2>&1
python3 F06_metrics.py > ../logs/F06_metrics.log 2>&1
echo DONE > ../logs/continue_done.flag
