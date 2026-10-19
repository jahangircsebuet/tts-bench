#!/usr/bin/env python3
"""Independent ASR validator (Whisper-large-v3-turbo) to check whether
MMS-TTS's quality validation, which uses MMS-ASR (same model family), is
optimistic from shared family-specific biases.

Whisper and MMS-ASR are architecturally unrelated (Whisper: encoder-decoder
transformer trained on web-scraped audio; MMS: wav2vec2-CTC trained on
religious-text recordings), so agreement between them is a much stronger
quality signal than either alone. Coverage caveat, reported explicitly
rather than hidden: Whisper's own documented language list covers only 100
languages total (ISO 639-1), and only a SUBSET of this benchmark's 76 audio
languages have a reliable ISO 639-3->639-1 mapping onto it (via `pycountry`,
not a hand-built guess) -- languages with no Whisper code are excluded from
this check and reported as a coverage gap, not silently treated as failures.

Transcribes a stratified ~1,200-file subsample (by tts_stratum x whether the
audio model supports that language) with Whisper, computes CER/WER (the
EXACT same Levenshtein-based method as full/code/F05_tts_quality.py) and
semantic similarity (the same sentence-transformers model) against the
original reference text, and compares the resulting quality decision to
MMS-ASR's on the identical files.

Run on GPU 2. Whisper is an encoder-decoder speech model, not an
autoregressive LLM -- it is loaded directly via `transformers`
(generate()-based inference), not vLLM, exactly as the project's own
MMS-TTS/MMS-ASR pipeline (full/code/F02, F05) already does for the same
reason; vLLM with the least memory needed is used for every LLM-judge task
in this folder (rubric_subset_reannotation_800.py,
other_safe_second_judge.py) but is not applicable to this encoder-decoder
ASR model.
"""
from __future__ import annotations
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import FULL, RESULTS, load_resource_tier_map, load_tts_quality, save

OUT = RESULTS / "independent_asr_validator"
OUT.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
SEM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STRATUM = {"mms": "native", "piper": "native", "mms_alias": "alias", "piper_alias": "proxy"}
N_TOTAL = 1200


# ---- identical normalization / error-rate / stratum functions to full/code/F05_tts_quality.py ----
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "S")))
    return re.sub(r"\s+", " ", s).strip()


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def error_rate(ref, hyp, unit):
    r = ref.split() if unit == "w" else list(ref.replace(" ", ""))
    h = hyp.split() if unit == "w" else list(hyp.replace(" ", ""))
    return levenshtein(r, h) / len(r) if r else float("nan")


def stratum_from_cer(c):
    if c is None or c != c:
        return "unscorable"
    return "good" if c <= 0.4 else "medium" if c <= 0.7 else "poor"


def whisper_lang_map(codes):
    """nllb_code -> Whisper 2-letter code, via pycountry's maintained ISO
    639-3 -> 639-1 table (not a hand-built guess), intersected with
    Whisper's own documented LANGUAGES dict."""
    import pycountry
    from transformers.models.whisper.tokenization_whisper import LANGUAGES as WHISPER_LANGS
    out = {}
    for code in codes:
        base = code.split("_")[0]
        lang = pycountry.languages.get(alpha_3=base)
        alpha2 = getattr(lang, "alpha_2", None) if lang else None
        if alpha2 and alpha2 in WHISPER_LANGS:
            out[code] = alpha2
    return out


def main():
    tq = load_tts_quality()
    tq["tts_stratum"] = tq["tts_route"].map(STRATUM)
    tier_map = load_resource_tier_map()
    tq["resource_tier"] = tq["nllb_code"].map(tier_map)

    lang_map = whisper_lang_map(tq["nllb_code"].unique())
    n_bench_langs = tq["nllb_code"].nunique()
    coverage = pd.DataFrame({"nllb_code": sorted(tq["nllb_code"].unique())})
    coverage["whisper_code"] = coverage["nllb_code"].map(lang_map)
    coverage["whisper_supported"] = coverage["whisper_code"].notna()
    save(coverage, OUT / "whisper_language_coverage.csv")
    print(f"Whisper language coverage: {coverage['whisper_supported'].sum()}/{n_bench_langs} "
          f"benchmark audio languages have a documented Whisper code "
          f"(via pycountry ISO 639-3->639-1, intersected with Whisper's own LANGUAGES dict).")

    tq["whisper_code"] = tq["nllb_code"].map(lang_map)
    covered = tq.dropna(subset=["whisper_code"]).copy()
    covered = covered[covered["recovered_transcript"].notna()]  # must have an audio_path to transcribe
    print(f"{len(covered):,} rows have Whisper-covered language AND an MMS-ASR row to compare against")

    rng = np.random.default_rng(0)
    parts = []
    for (stratum, tier), g in covered.groupby(["tts_stratum", "resource_tier"], dropna=False):
        n_take = max(1, round(N_TOTAL * len(g) / len(covered)))
        n_take = min(n_take, len(g))
        parts.append(g.loc[rng.choice(g.index, size=n_take, replace=False)])
    sample = pd.concat(parts, ignore_index=True).head(N_TOTAL)
    save(sample[["item_id", "nllb_code", "whisper_code", "tts_stratum", "resource_tier", "tts_route"]],
         OUT / "sampled_subset_index.csv")
    print(f"Whisper transcription sample: {len(sample):,} files")

    # ---- need audio_path + reference_text; tts_quality.jsonl has reference_text already ----
    import json
    meta = {}
    with open(FULL / "data" / "tts_metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta[r["item_id"]] = r.get("audio_path")
    sample["audio_path"] = sample["item_id"].map(meta)
    sample = sample.dropna(subset=["audio_path"])

    import torch
    torch.backends.cudnn.enabled = False  # this environment's cuDNN build throws
    # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on conv layers (documented in
    # full/README.md for MMS-TTS/MMS-ASR/Qwen2-Audio's own conv encoders);
    # same root cause, same fix: disable cuDNN and run the conv-heavy encoder
    # in fp32 rather than fp16.
    import scipy.io.wavfile
    import scipy.signal
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from sentence_transformers import SentenceTransformer

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES=2 remaps this to the physical GPU 2
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
    sem_model = SentenceTransformer(SEM_MODEL, device=device)

    results = []
    for i, row in enumerate(sample.to_dict("records")):
        wav_path = FULL / row["audio_path"]
        if not wav_path.exists():
            continue
        sr, audio = scipy.io.wavfile.read(wav_path)
        # MMS-TTS wavs are written as float32 already in [-1, 1] (VITS output);
        # Piper wavs are int16 PCM. Only normalize the int16 case -- dividing an
        # already-normalized float32 signal by 32768 again produces near-silent
        # audio and makes Whisper hallucinate generic fillers ("Thank you.").
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        if sr != 16000:
            # Piper audio is 22.05kHz; resample to 16kHz for Whisper, same as
            # full/code/F05_tts_quality.py already does for MMS-ASR.
            import math
            g = math.gcd(sr, 16000)
            audio = scipy.signal.resample_poly(audio, 16000 // g, sr // g)
            sr = 16000
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, torch.float32)
        with torch.no_grad():
            pred_ids = model.generate(inputs, language=row["whisper_code"], task="transcribe",
                                      max_new_tokens=200, condition_on_prev_tokens=False,
                                      temperature=0.0, no_repeat_ngram_size=3)
        transcript = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        results.append({"item_id": row["item_id"], "nllb_code": row["nllb_code"],
                        "tts_stratum": row["tts_stratum"], "resource_tier": row["resource_tier"],
                        "whisper_transcript": transcript})
        if (i + 1) % 100 == 0:
            print(f"  transcribed {i + 1}/{len(sample)}", flush=True)

    whisper_df = pd.DataFrame(results)
    ref_map = covered.set_index("item_id")["reference_text"]
    whisper_df["reference_text"] = whisper_df["item_id"].map(ref_map)
    whisper_df = whisper_df.dropna(subset=["reference_text"])

    wer_list, cer_list, sem_list = [], [], []
    ref_norms = whisper_df["reference_text"].map(norm).tolist()
    hyp_norms = whisper_df["whisper_transcript"].map(norm).tolist()
    for r, h in zip(ref_norms, hyp_norms):
        wer_list.append(error_rate(r, h, "w"))
        cer_list.append(error_rate(r, h, "c"))
    emb_ref = sem_model.encode(whisper_df["reference_text"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    emb_hyp = sem_model.encode(whisper_df["whisper_transcript"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    sem = (emb_ref * emb_hyp).sum(axis=1) / (np.linalg.norm(emb_ref, axis=1) * np.linalg.norm(emb_hyp, axis=1) + 1e-9)

    whisper_df["whisper_wer"] = wer_list
    whisper_df["whisper_cer"] = cer_list
    whisper_df["whisper_semantic_similarity"] = sem
    whisper_df["whisper_quality_stratum"] = whisper_df["whisper_cer"].map(stratum_from_cer)
    save(whisper_df, OUT / "whisper_transcription_results.csv")

    # ---- compare to MMS-ASR on the same items ----
    mms = covered.set_index("item_id")[["cer", "wer", "semantic_similarity", "quality_stratum"]]
    cmp = whisper_df.join(mms, on="item_id", rsuffix="_mms")
    cmp["stratum_agree"] = cmp["whisper_quality_stratum"] == cmp["quality_stratum"]
    save(cmp, OUT / "mms_vs_whisper_comparison.csv")

    agree_overall = cmp["stratum_agree"].mean() * 100
    corr_cer = cmp[["whisper_cer", "cer"]].dropna().corr().iloc[0, 1]
    print(f"\n== MMS-ASR vs. Whisper quality-stratum agreement: {agree_overall:.2f}% "
          f"(n={len(cmp):,}); CER correlation r={corr_cer:.3f} ==")

    by_stratum_route = cmp.groupby(["tts_stratum", "resource_tier"]).agg(
        n=("item_id", "size"),
        mms_good_pct=("quality_stratum", lambda s: 100 * (s == "good").mean()),
        whisper_good_pct=("whisper_quality_stratum", lambda s: 100 * (s == "good").mean()),
        stratum_agreement_pct=("stratum_agree", lambda s: 100 * s.mean()),
        mean_cer_mms=("cer", "mean"), mean_cer_whisper=("whisper_cer", "mean"),
    ).round(2).reset_index()
    save(by_stratum_route, OUT / "quality_comparison_by_stratum_tier.csv")
    print(by_stratum_route.to_string(index=False))

    (OUT / "summary.txt").write_text(
        f"Whisper language coverage: {coverage['whisper_supported'].sum()}/{n_bench_langs} languages\n"
        f"Sample transcribed: {len(sample):,}\n"
        f"MMS-ASR vs Whisper quality-stratum agreement: {agree_overall:.2f}%\n"
        f"CER correlation (MMS-ASR vs Whisper): r={corr_cer:.3f}\n"
    )


if __name__ == "__main__":
    main()
