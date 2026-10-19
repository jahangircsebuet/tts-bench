#!/usr/bin/env python3
"""Separate native TTS, same-language/macro-language alias TTS, and
cross-language proxy-voice TTS into three explicit strata instead of pooling
them into one "speech" condition.

  native TTS  = tts_route in {mms, piper}      (a genuine voice for that language)
  alias TTS   = tts_route == mms_alias          (shared macro-language checkpoint,
                                                  e.g. one Arabic/Persian voice
                                                  standing in for several
                                                  individual languages)
  proxy TTS   = tts_route == piper_alias        (a DIFFERENT, related language's
                                                  voice standing in for the
                                                  target language, e.g. a
                                                  Spanish/Portuguese voice for
                                                  Asturian/Galician/Occitan)

Native is reported as the PRIMARY benchmark result; alias and proxy are
reported separately and labeled diagnostic/exploratory.
CPU-only.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_master, rate, save, wilson

OUT = RESULTS / "tts_route_stratification"

STRATUM = {"mms": "native", "piper": "native", "mms_alias": "alias", "piper_alias": "proxy"}


def agg(df, keys):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        u = g["unsafe"].dropna()
        lo, hi = wilson(u.sum(), len(u))
        rows.append({**dict(zip(keys, k)), "n": len(g), "n_scored": len(u),
                     "UCR_pct": rate(g["unsafe"]), "UCR_lo": lo, "UCR_hi": hi,
                     "RR_pct": rate(g["refusal"]), "CR_pct": rate(g["comp"])})
    return pd.DataFrame(rows).round(2)


def main():
    master = load_master()
    sp = master[master["modality"] == "speech"].copy()
    sp["tts_stratum"] = sp["tts_route"].map(STRATUM)
    assert sp["tts_stratum"].notna().all(), "unmapped tts_route values present"

    full = agg(sp, ["audio_model", "condition", "tts_stratum"])
    save(full, OUT / "all_strata_combined.csv")

    native = full[full["tts_stratum"] == "native"].drop(columns=["tts_stratum"])
    alias = full[full["tts_stratum"] == "alias"].drop(columns=["tts_stratum"])
    proxy = full[full["tts_stratum"] == "proxy"].drop(columns=["tts_stratum"])
    save(native, OUT / "native_PRIMARY.csv")
    save(alias, OUT / "alias_DIAGNOSTIC.csv")
    save(proxy, OUT / "proxy_DIAGNOSTIC.csv")

    # Per-language-within-stratum breakdown, so e.g. "native" isn't itself
    # hiding an mms-vs-piper sub-effect.
    by_route = agg(sp, ["audio_model", "condition", "tts_route"])
    save(by_route, OUT / "by_raw_tts_route_for_reference.csv")

    # Headline-relevant slice: speech_lang only, native vs alias vs proxy side by side.
    sl = full[full["condition"] == "speech_lang"].pivot(index="audio_model", columns="tts_stratum",
                                                         values=["n", "UCR_pct", "CR_pct"])
    sl.to_csv(OUT / "speech_lang_native_vs_alias_vs_proxy_pivot.csv")
    print(f"[OK] results/{OUT.name}/speech_lang_native_vs_alias_vs_proxy_pivot.csv")

    print("\n== TTS route stratification summary (speech_lang, CR% by stratum) ==")
    print(full[full["condition"] == "speech_lang"][["audio_model", "tts_stratum", "n", "UCR_pct", "CR_pct"]]
          .to_string(index=False))
    print("\nNote: 'native' is the primary benchmark result; 'alias' and 'proxy'")
    print("rows measure voice-substitution artifacts as much as target-language")
    print("model behavior and should be read as diagnostic, not headline, numbers.")


if __name__ == "__main__":
    main()
