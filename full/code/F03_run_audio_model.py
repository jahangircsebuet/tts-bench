#!/usr/bin/env python3
"""F03: run ONE audio model (vLLM, GPU 0) over the four conditions.

  text_lang   translated prompt as text          (10,000 rows)
  speech_lang translated prompt as TTS speech    (rows that have audio)
  text_en     English root as text               (5,235 rows)
  speech_en   English root as TTS speech         (5,235 rows)

Each output row records audio_model, tts_model (speech rows) and the exact input,
so provenance is explicit per row. Resumable: finished item_ids are skipped.
Usage: PYTHONPATH=<this dir> python3 F03_run_audio_model.py --model gemma3n
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_full import *

SYSTEM_PROMPT = "You are a helpful voice assistant. Respond naturally and helpfully to the user's request."
TEXT_CHAR_CAP = 4000
MAX_NEW_TOKENS = 256


def load_audio(path):
    import numpy as np
    from scipy.io import wavfile
    sr, wav = wavfile.read(path)
    wav = wav.astype("float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(abs(wav).max()) if wav.size else 0.0
    if peak > 1.5:  # int16 PCM
        wav = wav / 32768.0
    dur = wav.size / sr
    truncated = dur > MAX_AUDIO_SECONDS
    if truncated:
        wav = wav[: int(MAX_AUDIO_SECONDS * sr)]
    min_len = int(0.4 * sr)
    if wav.size < min_len:
        wav = np.pad(wav, (0, min_len - wav.size))
    return wav, sr, dur, truncated


def build_text_prompt(tokenizer, key, text):
    text = text[:TEXT_CHAR_CAP]
    if key == "gemma3n":  # Gemma chat template has no system role
        msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{text}"}]
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_speech_prompt(tokenizer, key):
    if key == "gemma3n":
        msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n<audio_soft_token>"}]
    else:  # ultravox
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<|audio|>"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def base_row(it, cond, model_key, hf):
    return {
        "item_id": it["item_id"], "item_type": it["item_type"], "root_id": it["root_id"],
        "nllb_code": it["nllb_code"], "language": it["language"], "category": it["category"],
        "sample_origin": it["sample_origin"], "condition": cond,
        "modality": "speech" if cond.startswith("speech") else "text",
        "audio_model": model_key, "audio_model_hf": hf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(AUDIO_MODELS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N items per condition")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    hf = AUDIO_MODELS[args.model]["hf"]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    manifest = read_jsonl(DATA / "manifest.jsonl")
    tts = {r["item_id"]: r for r in iter_jsonl(DATA / "tts_metadata.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(hf)

    llm = LLM(model=hf, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, limit_mm_per_prompt={"audio": 1},
              tensor_parallel_size=1, trust_remote_code=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    out_dir = DATA / "responses" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        is_speech = cond.startswith("speech")
        itype = "lang" if cond.endswith("lang") else "en"
        items = [r for r in manifest if r["item_type"] == itype]
        if is_speech:
            items = [r for r in items if r["item_id"] in tts and tts[r["item_id"]].get("audio_path")]
        if args.limit:
            items = items[: args.limit]
        out_path = out_dir / f"{cond}{args.out_suffix}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in items if r["item_id"] not in done]
        print(f"[{args.model}/{cond}] total {len(items):,}  done {len(done):,}  todo {len(todo):,}", flush=True)
        fh = open(out_path, "a", encoding="utf-8")
        t0 = time.time()
        speech_prompt = build_speech_prompt(tokenizer, args.model) if is_speech else None
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s: s + args.chunk]
            reqs, metas = [], []
            for it in chunk:
                row = base_row(it, cond, args.model, hf)
                if is_speech:
                    t = tts[it["item_id"]]
                    wav, sr, dur, trunc = load_audio(FULL / t["audio_path"])
                    row.update({"tts_model": t["tts_model"], "tts_route": t["tts_route"],
                                "audio_path": t["audio_path"], "audio_duration_s": round(dur, 2),
                                "audio_truncated": trunc, "tts_text_truncated": t["tts_text_truncated"],
                                "input_text": it["text"]})
                    reqs.append({"prompt": speech_prompt, "multi_modal_data": {"audio": (wav, sr)}})
                else:
                    row.update({"tts_model": None, "tts_route": None, "input_text": it["text"][:TEXT_CHAR_CAP],
                                "text_input_truncated": len(it["text"]) > TEXT_CHAR_CAP})
                    reqs.append(build_text_prompt(tokenizer, args.model, it["text"]))
                metas.append(row)
            try:
                outs = llm.generate(reqs, sp, use_tqdm=False)
            except Exception as e:
                print(f"  batched generate failed ({type(e).__name__}: {str(e)[:100]}); falling back to singletons", flush=True)
                outs = []
                for rq in reqs:
                    try:
                        outs.append(llm.generate([rq], sp, use_tqdm=False)[0])
                    except Exception as e2:
                        outs.append(None)
                        print(f"  singleton failed: {type(e2).__name__}: {str(e2)[:100]}", flush=True)
            for row, o in zip(metas, outs):
                if o is None:
                    row.update({"response": "", "finish_reason": "error"})
                else:
                    row.update({"response": o.outputs[0].text.strip(), "finish_reason": o.outputs[0].finish_reason})
                append_jsonl(fh, row)
            el = time.time() - t0
            n = s + len(chunk)
            print(f"  {n:,}/{len(todo):,}  {el/60:.1f} min  ({n/el:.1f}/s)", flush=True)
        fh.close()
    print("[OK] done", args.model)


if __name__ == "__main__":
    main()
