"""Shared paths, model registry and IO helpers for the full 10k-benchmark run."""
from __future__ import annotations
import json
from pathlib import Path

FULL = Path(__file__).resolve().parent.parent
ROOT = FULL.parent  # repository root
DATA = FULL / "data"
RESULTS = FULL / "results"
LOGS = FULL / "logs"

# BENCH_10K and ROOT_TEXT_FILES point to the upstream prompt-curation and
# stratified-sampling pipeline that produced the 10,000-row manifest this
# benchmark starts from -- that pipeline is a separate, much larger project
# and is not part of this package. They are used only by
# F01_build_manifest.py, to document how the manifest was originally built;
# the manifest it produces (data/manifest.jsonl) is included directly, so
# nothing downstream of F01 depends on these paths existing.
BENCH_10K = ROOT / "upstream_data_not_included/overall_stratified_10k.jsonl"
ROOT_TEXT_FILES = [ROOT / f"upstream_data_not_included/stage06_unsafe_roots_part_{i:02d}.jsonl" for i in range(1, 7)]
MMS_SUPPORT_JSON = ROOT / "config" / "mms_tts_support_full.json"

TTS_CHAR_CAP = 500          # text longer than this is cut before TTS (flagged in manifest)
MAX_AUDIO_SECONDS = 29.5    # audio longer than this is cut before the audio model (flagged per response)

# audio models under test; `family` is used for the leave-own-family-out judge rule
AUDIO_MODELS = {
    "gemma3n":  {"hf": "google/gemma-3n-E4B-it",                      "family": "google"},
    "ultravox": {"hf": "fixie-ai/ultravox-v0_6-llama-3_1-8b",         "family": "meta"},
}
JUDGES = {
    "gemma3_12b": {"hf": "google/gemma-3-12b-it",       "family": "google", "gpu_mem": 0.4},
    "qwen3_14b":  {"hf": "Qwen/Qwen3-14B",              "family": "alibaba", "gpu_mem": 0.5},
    "aya_8b":     {"hf": "CohereLabs/aya-expanse-8b",   "family": "cohere", "gpu_mem": 0.3},
}
CONDITIONS = ["text_lang", "speech_lang", "text_en", "speech_en"]


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path):
    return list(iter_jsonl(path))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows):,} rows -> {path}")


def append_jsonl(fh, row):
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
