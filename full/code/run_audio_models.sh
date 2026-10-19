#!/bin/bash
# audio models one by one on GPU 0; resumable
cd "$(dirname "$0")"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
python3 F03_run_audio_model.py --model gemma3n  --gpu-memory-utilization 0.4 > ../logs/F03_gemma3n.log 2>&1
python3 F03_run_audio_model.py --model ultravox --gpu-memory-utilization 0.3 > ../logs/F03_ultravox.log 2>&1
echo ALL_AUDIO_DONE > ../logs/audio_done.flag
