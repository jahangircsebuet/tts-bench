#!/bin/bash
# waits for audio models + Piper TTS, then: supplemental speech pass for Piper rows -> judges one by one -> TTS quality -> metrics
cd "$(dirname "$0")"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
while [ ! -f ../logs/audio_done.flag ] || pgrep -f F02c_generate_tts_piper > /dev/null; do sleep 20; done
python3 F03_run_audio_model.py --model gemma3n  --gpu-memory-utilization 0.4 --conditions speech_lang > ../logs/F03b_gemma3n_piper.log 2>&1
python3 F03_run_audio_model.py --model ultravox --gpu-memory-utilization 0.3 --conditions speech_lang > ../logs/F03b_ultravox_piper.log 2>&1
for j in gemma3_12b qwen3_14b aya_8b; do
  python3 F04_judge_responses.py --judge $j > ../logs/F04_$j.log 2>&1
done
python3 F05_tts_quality.py > ../logs/F05_tts_quality.log 2>&1
python3 F06_metrics.py > ../logs/F06_metrics.log 2>&1
echo ALL_DONE > ../logs/all_done.flag
