#!/usr/bin/env python3
"""Document, per audio model, which of the benchmark's 100 languages are
OFFICIALLY claimed as supported by that model's own public documentation --
as opposed to being evaluated zero-shot/unsupported -- and recompute results
split by that grouping.

Sourcing (see SOURCES below for exact quotes/URLs):
  - Ultravox v0.6 (fixie-ai/ultravox-v0_6-llama-3_1-8b): the fixie.ai v0.5
    announcement blog enumerates an exact 42-language list, and the v0.6
    model card carries the "42 languages" tag, so we treat that list as
    Ultravox's officially documented audio-language support.
  - Gemma-3n (google/gemma-3n-E4B-it): Google's own developer guide states
    "140 languages for text and multimodal understanding of 35 languages" but
    -- we checked ai.google.dev/gemma/docs/gemma-3n, the HF model card, and
    the developer-guide blog post directly -- Google does NOT publish an
    enumerated list of which 35 languages those are. The only concrete,
    named claim is "strong Automatic Speech Translation results ... between
    English and Spanish, French, Italian, and Portuguese." We therefore mark
    ONLY {English, Spanish, French, Italian, Portuguese} as "documented" for
    Gemma-3n's audio/multimodal pathway, and every other language as
    "undocumented" (explicitly NOT the same as "unsupported" -- Google claims
    35 multimodal languages exist, just does not say which) rather than
    silently assuming parity with the 140-language TEXT figure, which Google
    explicitly scopes to text only.

This is an important finding in itself: the third-party open adapter
(Ultravox) publishes a more specific, checkable language-support claim than
the vendor-released natively-multimodal model (Gemma-3n) does for its own
audio pathway.

CPU-only; no GPU/vLLM needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import FULL, RESULTS, load_master, rate, save, wilson

OUT = RESULTS / "documented_language_support"

SOURCES = """\
# Sources (fetched/quoted 2026-10-12)

## Ultravox (fixie-ai/ultravox-v0_6-llama-3_1-8b)
- https://www.ultravox.ai/blog/ultravox-v0-5-taking-the-lead-in-speech-understanding
  "Arabic, Belarusian, Bengali, Bulgarian, Chinese, Czech, Danish, Dutch,
   English, Estonian, Finnish, French, Galician, Georgian, German, Greek,
   Hindi, Hungarian, Italian, Japanese, Latvian, Lithuanian, Macedonian,
   Marathi, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak,
   Slovenian, Spanish, Swahili, Swedish, Tamil, Thai, Turkish, Ukrainian,
   Urdu, Vietnamese, Welsh." (42 languages)
- https://huggingface.co/fixie-ai/ultravox-v0_6-llama-3_1-8b carries the
  "42 languages" tag; no further per-checkpoint enumeration is given on that
  page, so we carry forward the v0.5 list as the documented set for v0.6.

## Gemma-3n (google/gemma-3n-E4B-it)
- https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
  "supporting 140 languages for text and multimodal understanding of
   35 languages" -- count given, NO enumerated list.
  "particularly strong results for audio speech translation between English
   and Spanish, French, Italian, and Portuguese" -- the only NAMED languages
   for the audio pathway specifically.
- https://ai.google.dev/gemma/docs/gemma-3n and
  https://huggingface.co/google/gemma-3n-E4B-it: neither enumerates the
  35-language multimodal list either; both only repeat the 140-language
  TEXT-training-data figure.

Conclusion: we can mark Ultravox's support status for all 100 benchmark
languages with a documented source; for Gemma-3n we can only mark 5 languages
(English + the 4 named AST languages) as documented, and must mark the
remaining 95 as "undocumented" (NOT "unsupported" -- Google does not say).
"""

# --- Ultravox's documented 42 languages (English names, as published) -----
ULTRAVOX_42 = {
    "Arabic", "Belarusian", "Bengali", "Bulgarian", "Chinese", "Czech", "Danish",
    "Dutch", "English", "Estonian", "Finnish", "French", "Galician", "Georgian",
    "German", "Greek", "Hindi", "Hungarian", "Italian", "Japanese", "Latvian",
    "Lithuanian", "Macedonian", "Marathi", "Persian", "Polish", "Portuguese",
    "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Spanish",
    "Swahili", "Swedish", "Tamil", "Thai", "Turkish", "Ukrainian", "Urdu",
    "Vietnamese", "Welsh",
}

GEMMA3N_DOCUMENTED = {"English", "Spanish", "French", "Italian", "Portuguese"}

# benchmark language name -> the name used in the official lists above, where
# they differ (e.g. our benchmark has two Chinese scripts; Ultravox's list
# just says "Chinese").
NAME_ALIAS = {
    "Chinese (Simplified)": "Chinese",
    "Chinese (Traditional)": "Chinese",
    "Filipino (Tagalog)": "Filipino",
    "Oriya (Odia)": "Odia",
}


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
    (OUT / "SOURCES.md").write_text(SOURCES)

    t2 = pd.read_csv(FULL / "results" / "T2_per_language.csv")
    langs = t2[["language", "nllb_code"]].drop_duplicates().sort_values("language").reset_index(drop=True)
    langs["lookup_name"] = langs["language"].map(lambda x: NAME_ALIAS.get(x, x))

    langs["ultravox_documented"] = langs["lookup_name"].isin(ULTRAVOX_42)
    langs["gemma3n_documented"] = langs["lookup_name"].isin(GEMMA3N_DOCUMENTED)
    save(langs, OUT / "language_support_table.csv")

    print(f"Ultravox: {langs['ultravox_documented'].sum()}/100 benchmark languages are in its documented 42-language list.")
    print(f"Gemma-3n: {langs['gemma3n_documented'].sum()}/100 benchmark languages are in its documented (named) audio-support list.")

    master = load_master()
    sl = master[(master["item_type"] == "lang") & (master["condition"] == "speech_lang")].copy()
    support_map = {"ultravox": dict(zip(langs["language"], langs["ultravox_documented"])),
                   "gemma3n": dict(zip(langs["language"], langs["gemma3n_documented"]))}
    sl["documented_support"] = sl.apply(lambda r: support_map[r["audio_model"]].get(r["language"], False), axis=1)
    sl["support_group"] = sl["documented_support"].map({True: "documented_supported", False: "undocumented"})

    by_support = agg(sl, ["audio_model", "support_group"])
    save(by_support, OUT / "speech_lang_results_by_support_group.csv")

    print("\n== documented-language-support summary: CR% in speech_lang, by group ==")
    print(by_support.to_string(index=False))
    print("\nCaveat for Gemma-3n: 'undocumented' here means Google does not name")
    print("the language as audio-supported, NOT that Google claims it is unsupported")
    print("-- Google states 35 multimodal languages exist but never lists them.")


if __name__ == "__main__":
    main()
