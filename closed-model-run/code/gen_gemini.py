#!/usr/bin/env python3
"""Generate responses from Gemini over the unsafe or benign dataset, at the
SAME size as the open-model runs (reusing existing audio).

Model id note: the user asked for "Gemini 2.5 Flash-Lite"
(`models/gemini-2.5-flash-lite`), but that id now 404s ("no longer available
to new users ... use models/gemini-3.5-flash-lite"), confirmed via a real API
call against this account. `models/gemini-3.5-flash-lite` is used instead --
Google's own suggested, same-tier replacement for the exact model requested.

No hard budget cap was set for this model (the user's $100 cap was stated
specifically for the OpenAI model), but cost is tracked anyway for
transparency (see Budget class) since real money is still being spent;
Gemini's own pricing page puts this model at $0.10/1M text input, $0.30/1M
audio input, $0.40/1M output (standard tier) -- real test calls confirmed
full responses complete cleanly (finish_reason=STOP) well within a 256-token
cap (the same cap used for every other model in this study), so no special
reasoning/output-cap tuning was needed here.

Usage:
  python3 gen_gemini.py --dataset unsafe --conditions text_en speech_en
  python3 gen_gemini.py --dataset benign
Requires GEMINI_API_KEY in the environment (see run_closed.sh, which sources
TTS/.env).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_closed import *

MODEL = "models/gemini-3.5-flash-lite"
MAX_OUTPUT_TOKENS = 256
PRICE = {"text_in": 0.10, "audio_in": 0.30, "out": 0.40}  # $/1M tokens, standard tier


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.spent = 0.0
        if self.path.exists():
            self.spent = json.loads(self.path.read_text()).get("spent_usd", 0.0)

    def add(self, usage_metadata):
        if not usage_metadata:
            return 0.0
        audio_tok, text_tok = 0, 0
        for d in (usage_metadata.prompt_tokens_details or []):
            if str(d.modality).endswith("AUDIO"):
                audio_tok += d.token_count or 0
            else:
                text_tok += d.token_count or 0
        out_tok = usage_metadata.candidates_token_count or 0
        cost = audio_tok / 1e6 * PRICE["audio_in"] + text_tok / 1e6 * PRICE["text_in"] + out_tok / 1e6 * PRICE["out"]
        self.spent += cost
        self.path.write_text(json.dumps({"spent_usd": round(self.spent, 4)}))
        return cost


def call_one(wav_bytes_or_text, is_speech, max_retries=4):
    import os
    from google import genai
    from google.genai import types

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            parts = [types.Part(text=SYSTEM_PROMPT)]
            if is_speech:
                parts.append(types.Part.from_bytes(data=wav_bytes_or_text, mime_type="audio/wav"))
            else:
                parts.append(types.Part(text=wav_bytes_or_text))
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.0),
            )
            text = resp.text or ""
            cand = resp.candidates[0] if resp.candidates else None
            status = str(cand.finish_reason) if cand else None
            return text.strip(), resp.usage_metadata, status
        except Exception as e:
            if attempt == max_retries - 1:
                return "", None, f"error:{type(e).__name__}:{str(e)[:150]}"
            time.sleep(2 ** attempt)


def base_row(it, cond, text_en):
    return {
        "item_id": it["item_id"], "root_id": it["root_id"], "item_type": it["item_type"],
        "nllb_code": it["nllb_code"], "language": it["language"],
        "condition": cond, "modality": "speech" if cond.startswith("speech") else "text",
        "closed_model": "gemini_flash_lite", "closed_model_id": MODEL, "text_en": text_en,
    }


def usage_to_dict(u):
    if u is None:
        return None
    return {
        "total_token_count": u.total_token_count, "candidates_token_count": u.candidates_token_count,
        "prompt_token_count": u.prompt_token_count,
        "prompt_tokens_details": [{"modality": str(d.modality), "token_count": d.token_count}
                                   for d in (u.prompt_tokens_details or [])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unsafe", "benign"])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = RESULTS / "gemini_flash_lite" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(RESULTS / "gemini_flash_lite" / "spend.json")

    if args.dataset == "unsafe":
        manifest, tts = load_unsafe_manifest()
    else:
        man_df = load_benign_manifest()

    for cond in args.conditions:
        is_speech = cond.startswith("speech")
        if args.dataset == "unsafe":
            items = unsafe_items_for_condition(manifest, tts, cond)
        else:
            items = benign_items_for_condition(man_df, cond)
        if args.limit:
            items = items[: args.limit]

        out_path = out_dir / f"{cond}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in items if r["item_id"] not in done]
        print(f"[gemini/{args.dataset}/{cond}] total {len(items):,} done {len(done):,} todo {len(todo):,} "
              f"spent_so_far=${budget.spent:.4f}", flush=True)

        fh = open(out_path, "a", encoding="utf-8")
        t0 = time.time()

        def prep(it):
            if is_speech:
                t = tts[it["item_id"]] if args.dataset == "unsafe" else it
                path = unsafe_audio_path(t) if args.dataset == "unsafe" else benign_audio_path(t)
                wav, sr, dur, trunc = load_audio_float(path)
                payload = wav_bytes_16bit(wav, sr)
                return it, payload, True, dur, trunc
            else:
                return it, it["text"][:TEXT_CHAR_CAP], False, None, False

        n_done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {}
            for it in todo:
                item, payload, is_sp, dur, trunc = prep(it)
                fut = ex.submit(call_one, payload, is_sp)
                futs[fut] = (item, is_sp, dur, trunc)
            for fut in as_completed(futs):
                item, is_sp, dur, trunc = futs[fut]
                text, usage, status = fut.result()
                cost = budget.add(usage)
                row = base_row(item, cond, item.get("text_en", item.get("text")))
                if is_sp:
                    row.update({"audio_duration_s": round(dur, 2) if dur else None, "audio_truncated": trunc})
                else:
                    row.update({"input_text": item["text"][:TEXT_CHAR_CAP]})
                row.update({"response": text, "status": status, "usage": usage_to_dict(usage),
                            "cost_usd": round(cost, 6) if cost else 0.0})
                append_jsonl(fh, row)
                n_done += 1
                if n_done % 500 == 0:
                    el = time.time() - t0
                    print(f"  {n_done:,}/{len(todo):,}  {el/60:.1f} min  spent=${budget.spent:.4f}", flush=True)
        fh.close()
    print(f"[OK] done gemini_flash_lite/{args.dataset}  total_spent=${budget.spent:.4f}")


if __name__ == "__main__":
    main()
