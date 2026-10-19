#!/usr/bin/env python3
"""F02c (run with .venv_piper, CPU): synthesize the Piper-routed manifest rows.
Same output layout/provenance fields as F02; adds asr_lang for the round-trip check. Resumable."""
from __future__ import annotations
import json, sys, time, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_full import *
from piper import PiperVoice


def main():
    plan = json.load(open(DATA / "piper_plan.json"))
    manifest = read_jsonl(DATA / "manifest.jsonl")
    meta_path = DATA / "tts_metadata.jsonl"
    done = {r["item_id"] for r in iter_jsonl(meta_path)} if meta_path.exists() else set()
    by_code = {}
    for r in manifest:
        if r["tts_route"] in ("piper", "piper_alias") and r["item_id"] not in done:
            by_code.setdefault(r["nllb_code"], []).append(r)
    total = sum(map(len, by_code.values()))
    print(f"to synthesize: {total:,} across {len(by_code)} languages", flush=True)
    fh = open(meta_path, "a", encoding="utf-8")
    t0, n = time.time(), 0
    for code, items in sorted(by_code.items()):
        p = plan[code]
        try:
            voice = PiperVoice.load(p["onnx"], config_path=p["config"])
        except Exception as e:
            print(f"[{code}] voice {p['voice']} failed to load ({type(e).__name__}: {str(e)[:80]}); skipping", flush=True)
            for r in items:
                append_jsonl(fh, {"item_id": r["item_id"], "item_type": r["item_type"], "root_id": r["root_id"],
                                  "nllb_code": code, "language": r["language"], "tts_model": p["tts_model"],
                                  "tts_route": r["tts_route"], "status": f"error: voice load failed: {type(e).__name__}",
                                  "audio_path": None, "duration_s": 0.0, "asr_lang": p["asr_lang"]})
            continue
        sr = voice.config.sample_rate
        out_dir = DATA / "audio" / ("piper-" + p["voice"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in items:
            text = r["text"][:TTS_CHAR_CAP]
            path = out_dir / f"{r['item_id'].replace('::', '__')}.wav"
            try:
                with wave.open(str(path), "wb") as w:
                    voice.synthesize_wav(text, w)
                with wave.open(str(path), "rb") as w:
                    dur = w.getnframes() / w.getframerate()
                status = "ok" if dur > 0 else "error: empty waveform"
            except Exception as e:
                status, dur = f"error: {type(e).__name__}: {str(e)[:120]}", 0.0
            rec = {"item_id": r["item_id"], "item_type": r["item_type"], "root_id": r["root_id"],
                   "nllb_code": r["nllb_code"], "language": r["language"], "tts_model": p["tts_model"],
                   "tts_route": r["tts_route"], "speaker": "default", "sample_rate": sr,
                   "tts_text_truncated": r["tts_text_truncated"], "text_used_for_tts": text, "status": status,
                   "asr_lang": p["asr_lang"],
                   "audio_path": str(path.relative_to(FULL)) if status == "ok" else None,
                   "duration_s": round(dur, 3)}
            append_jsonl(fh, rec)
            n += 1
            if n % 200 == 0:
                print(f"  {n:,}/{total:,}  {(time.time()-t0)/60:.1f} min", flush=True)
        print(f"[{code}] {p['voice']} done {len(items)}", flush=True)
    print(f"[OK] {n:,} items in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
