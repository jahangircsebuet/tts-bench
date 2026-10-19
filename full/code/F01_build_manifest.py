#!/usr/bin/env python3
"""F01: build the master manifest for the full run.

One row per benchmark item. `item_type` is "lang" (the 10,000 translated rows of
overall_stratified_10k.jsonl: 6,180 old_reuse + 3,820 new_stratified) or "en"
(one English row per unique root, the cross-lingual baseline). Every row records
the TTS model assigned to it, so provenance is fixed here and carried through
every later stage.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_full import *


def tts_assignment(nllb_code, support):
    rec = support.get(nllb_code)
    if rec is None:
        return None, "none"
    if rec["status"] == "SUPPORTED":
        return rec["repo"], "mms"
    if rec["status"] == "SUPPORTED_VIA_ALIAS":
        return rec["macro_alias"], "mms_alias"
    return None, "none"


def main():
    support = {r["nllb_code"]: r for r in json.load(open(MMS_SUPPORT_JSON))}
    bench = read_jsonl(BENCH_10K)

    root_ids = {r["root_id"] for r in bench}
    en_text, en_meta = {}, {}
    for p in ROOT_TEXT_FILES:
        for r in iter_jsonl(p):
            if r["root_id"] in root_ids:
                en_text[r["root_id"]] = r["prompt"]
                en_meta[r["root_id"]] = r.get("prompt_label")
    missing = root_ids - set(en_text)
    print(f"roots: {len(root_ids):,}; English text found: {len(en_text):,}; missing: {len(missing)}")
    assert not missing, list(missing)[:5]
    print("English root label values:", Counter(en_meta.values()))

    rows = []
    for r in bench:
        repo, route = tts_assignment(r["nllb_code"], support)
        text = r["prompt_translated_lang"]
        rows.append({
            "item_id": r["sample_id"], "item_type": "lang", "root_id": r["root_id"],
            "nllb_code": r["nllb_code"], "language": r["language_name"], "category": r["category"],
            "sample_origin": r["sample_origin"],
            "text": text, "text_en": en_text[r["root_id"]],
            "tts_model": repo, "tts_route": route,
            "tts_text_truncated": len(text) > TTS_CHAR_CAP,
        })
    for rid in sorted(root_ids):
        cat = next(r["category"] for r in bench if r["root_id"] == rid)
        text = en_text[rid]
        rows.append({
            "item_id": f"en::{rid}", "item_type": "en", "root_id": rid,
            "nllb_code": "eng_Latn", "language": "English", "category": cat,
            "sample_origin": "english_root",
            "text": text, "text_en": text,
            "tts_model": "facebook/mms-tts-eng", "tts_route": "mms",
            "tts_text_truncated": len(text) > TTS_CHAR_CAP,
        })
    write_jsonl(DATA / "manifest.jsonl", rows)

    lang = [r for r in rows if r["item_type"] == "lang"]
    print("lang rows by TTS route:", Counter(r["tts_route"] for r in lang))
    print("lang rows by sample_origin:", Counter(r["sample_origin"] for r in lang))
    langs_by_route = {}
    for r in lang:
        langs_by_route.setdefault(r["tts_route"], set()).add(r["nllb_code"])
    print({k: len(v) for k, v in langs_by_route.items()})
    print("rows needing TTS truncation:", sum(r["tts_text_truncated"] for r in rows))


if __name__ == "__main__":
    main()
