#!/usr/bin/env python3
"""Dedicated language-ID validation on recovered ASR transcripts.

The pipeline never checked whether the MMS-ASR round-trip transcript is
actually IN the target language -- CER/WER/semantic-similarity all measure
"is this intelligible text close to the source text", not "is this the
right language". Given other_safe_subcategorization.py's finding that
LANGUAGE_MISMATCH is the dominant OTHER_SAFE subcategory, this script runs
GlotLID (cis-lmu/glotlid, ~2,000-language fastText LID model, already cited
in the paper's own bibliography as glotlid_kargaran2023) on every one of the
11,933 recovered MMS-ASR transcripts in tts_quality.jsonl, at FULL scale
(fastText inference is fast enough that no subsampling is needed), and
reports:
  1. correct-language rate by TTS stratum (native/alias/proxy) and by
     resource tier (High/Low, from config/language_metadata.py);
  2. a rerun of the headline speech_lang CR restricted to items whose
     recovered audio was independently confirmed to be in the target
     language AND synthesized via a native voice.

CPU-only (fastText is a small, fast linear classifier -- no GPU/vLLM
applies here; this is the direct text-based analogue of "LID on recovered
ASR transcripts" -- the raw-audio LID alternative would require an
additional audio-domain LID model and was not pursued given the
transcript-based check already directly measures the mechanism
other_safe_subcategorization.py flagged).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_glotlid, load_master, load_resource_tier_map, load_tts_quality, rate, save, wilson

OUT = RESULTS / "language_id_validation"
OUT.mkdir(parents=True, exist_ok=True)

STRATUM = {"mms": "native", "piper": "native", "mms_alias": "alias", "piper_alias": "proxy"}


def main():
    tq = load_tts_quality()
    tq["tts_stratum"] = tq["tts_route"].map(STRATUM)
    tier_map = load_resource_tier_map()
    tq["resource_tier"] = tq["nllb_code"].map(tier_map)

    model = load_glotlid()
    texts = tq["recovered_transcript"].fillna("").str.replace("\n", " ").str.strip()
    non_empty = texts != ""
    preds = pd.Series([None] * len(tq), index=tq.index, dtype=object)
    confs = pd.Series([float("nan")] * len(tq), index=tq.index)
    batch_texts = texts[non_empty].tolist()
    labels, probs = model.predict(batch_texts, k=1)
    preds.loc[non_empty] = [lb[0].replace("__label__", "") for lb in labels]
    confs.loc[non_empty] = [float(p[0]) for p in probs]
    tq["lid_predicted_code"] = preds
    tq["lid_confidence"] = confs

    # GlotLID's script-qualified codes should match nllb_code's own script suffix for an exact
    # match; also allow a script-insensitive match (same base ISO 639-3, different script tag)
    # as a secondary, looser "right language, GlotLID guessed a different script" category.
    tq["lid_exact_match"] = tq["lid_predicted_code"] == tq["nllb_code"]
    tq["target_base"] = tq["nllb_code"].str.split("_").str[0]
    tq["lid_base"] = tq["lid_predicted_code"].astype(str).str.split("_").str[0]
    tq["lid_base_match"] = tq["lid_base"] == tq["target_base"]
    tq["lid_scored"] = non_empty

    save(tq[["item_id", "nllb_code", "tts_route", "tts_stratum", "resource_tier", "recovered_transcript",
             "lid_predicted_code", "lid_confidence", "lid_exact_match", "lid_base_match"]],
         OUT / "lid_results_full.csv")

    # ---- 1. correct-language rate by stratum and resource tier ----
    def agg(df, keys):
        rows = []
        for k, g in df.groupby(keys, dropna=False):
            k = k if isinstance(k, tuple) else (k,)
            scored = g[g["lid_scored"]]
            rows.append({**dict(zip(keys, k)), "n": len(g), "n_scored": len(scored),
                         "exact_match_pct": round(100 * scored["lid_exact_match"].mean(), 2) if len(scored) else float("nan"),
                         "base_lang_match_pct": round(100 * scored["lid_base_match"].mean(), 2) if len(scored) else float("nan"),
                         "mean_lid_confidence": round(scored["lid_confidence"].mean(), 3) if len(scored) else float("nan")})
        return pd.DataFrame(rows)

    by_stratum = agg(tq, ["tts_stratum"])
    save(by_stratum, OUT / "correct_language_rate_by_stratum.csv")
    by_tier = agg(tq.dropna(subset=["resource_tier"]), ["tts_stratum", "resource_tier"])
    save(by_tier, OUT / "correct_language_rate_by_stratum_and_tier.csv")
    by_lang = agg(tq, ["nllb_code", "tts_stratum"])
    save(by_lang, OUT / "correct_language_rate_by_language.csv")

    print("\n== correct-language rate by TTS stratum ==")
    print(by_stratum.to_string(index=False))
    print("\n== by stratum x resource tier ==")
    print(by_tier.to_string(index=False))

    # ---- 2. rerun headline CR restricted to LID-confirmed native audio ----
    master = load_master()
    sl = master[(master["item_type"] == "lang") & (master["condition"] == "speech_lang")].copy()
    lid_flag = tq.set_index("item_id")[["lid_base_match", "tts_stratum"]]
    sl = sl.join(lid_flag, on="item_id", rsuffix="_lid")

    def ucr_cr(df):
        u = df["unsafe"].dropna()
        lo, hi = wilson(u.sum(), len(u))
        return pd.Series({"n": len(df), "n_scored": len(u), "UCR_pct": rate(df["unsafe"]), "UCR_lo": lo, "UCR_hi": hi,
                          "RR_pct": rate(df["refusal"]), "CR_pct": rate(df["comp"])})

    rows = []
    for am, g in sl.groupby("audio_model"):
        rows.append({"audio_model": am, "subset": "ALL speech_lang (original headline)", **ucr_cr(g).to_dict()})
        native = g[g["tts_stratum"] == "native"]
        rows.append({"audio_model": am, "subset": "native-route only (no LID filter)", **ucr_cr(native).to_dict()})
        native_lid_ok = native[native["lid_base_match"] == True]
        rows.append({"audio_model": am, "subset": "native-route AND LID-confirmed target language",
                     **ucr_cr(native_lid_ok).to_dict()})
    rerun_df = pd.DataFrame(rows).round(2)
    save(rerun_df, OUT / "headline_CR_rerun_on_lid_confirmed_native_audio.csv")

    print("\n== headline CR, original vs. native-only vs. native+LID-confirmed ==")
    print(rerun_df.to_string(index=False))


if __name__ == "__main__":
    main()
