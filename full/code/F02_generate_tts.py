#!/usr/bin/env python3
"""F02: text -> speech with Meta MMS-TTS (VITS), one checkpoint per language.

Reads data/manifest.jsonl, synthesizes every row that has a tts_model, writes
data/audio/<tts_model_name>/<item_id>.wav and appends one provenance record per
file to data/tts_metadata.jsonl. Resumable: rows already in tts_metadata.jsonl
are skipped. GPU 0, plain transformers (VITS is not a vLLM workload).
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_full import *


def seed_for(item_id):
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--limit-per-lang", type=int, default=0, help="debug: cap rows per TTS repo")
    args = ap.parse_args()

    import torch, scipy.io.wavfile
    torch.backends.cudnn.enabled = False  # env cuDNN sublibrary mismatch; see pilot README
    from transformers import VitsModel, AutoTokenizer
    device = f"cuda:{args.gpu_index}"

    manifest = read_jsonl(DATA / "manifest.jsonl")
    meta_path = DATA / "tts_metadata.jsonl"
    done = {r["item_id"] for r in iter_jsonl(meta_path)} if meta_path.exists() else set()
    print(f"already synthesized: {len(done):,}")

    by_repo = defaultdict(list)
    for r in manifest:
        if r["tts_model"] and r["item_id"] not in done:
            by_repo[r["tts_model"]].append(r)
    if args.limit_per_lang:
        by_repo = {k: v[: args.limit_per_lang] for k, v in by_repo.items()}
    total = sum(len(v) for v in by_repo.values())
    print(f"to synthesize: {total:,} items across {len(by_repo)} TTS checkpoints")

    uroman = None
    t0, n_done = time.time(), 0
    out_fh = open(meta_path, "a", encoding="utf-8")
    for repo, items in sorted(by_repo.items()):
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo).to(device).eval()
        sr = model.config.sampling_rate
        out_dir = DATA / "audio" / repo.split("/")[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in items:
            text = r["text"][:TTS_CHAR_CAP]
            used = text
            if getattr(tok, "is_uroman", False):
                if uroman is None:
                    import uroman as ur
                    uroman = ur.Uroman()
                used = uroman.romanize_string(text).lower()
            try:
                inputs = tok(used or ".", return_tensors="pt").to(device)
                if inputs["input_ids"].shape[1] == 0:
                    raise ValueError("empty token sequence")
                torch.manual_seed(seed_for(r["item_id"]))
                with torch.no_grad():
                    wav = model(**inputs).waveform.squeeze().float().cpu().numpy()
                status = "ok"
            except Exception as e:  # unusable text for this tokenizer
                wav, status = None, f"error: {type(e).__name__}: {str(e)[:120]}"
            rec = {
                "item_id": r["item_id"], "item_type": r["item_type"], "root_id": r["root_id"],
                "nllb_code": r["nllb_code"], "language": r["language"],
                "tts_model": repo, "tts_route": r["tts_route"], "speaker": "default",
                "sample_rate": sr, "tts_text_truncated": r["tts_text_truncated"],
                "text_used_for_tts": used, "status": status,
            }
            if wav is not None and wav.size > 0:
                path = out_dir / f"{r['item_id'].replace('::', '__')}.wav"
                scipy.io.wavfile.write(path, rate=sr, data=wav)
                rec["audio_path"] = str(path.relative_to(FULL))
                rec["duration_s"] = round(wav.size / sr, 3)
            else:
                rec["audio_path"], rec["duration_s"] = None, 0.0
                if status == "ok":
                    rec["status"] = "error: empty waveform"
            append_jsonl(out_fh, rec)
            n_done += 1
            if n_done % 500 == 0:
                el = time.time() - t0
                print(f"  {n_done:,}/{total:,}  {el/60:.1f} min  ({n_done/el:.1f}/s)", flush=True)
        del model
        torch.cuda.empty_cache()
        print(f"[{repo}] done {len(items)}", flush=True)
    out_fh.close()
    print(f"[OK] finished {n_done:,} items in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
