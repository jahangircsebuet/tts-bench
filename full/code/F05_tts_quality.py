#!/usr/bin/env python3
"""F05: TTS quality via ASR round trip (text -> MMS-TTS -> MMS-ASR -> text).

Per synthesized wav: transcribe with facebook/mms-1b-all (language adapter chosen from the
TTS checkpoint's language code), compare with the source text after lowercasing and punctuation
stripping, and report WER, CER, multilingual semantic similarity and a good/medium/poor stratum.
Carries tts_model / tts_route so quality can be reported per TTS model.
"""
from __future__ import annotations
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_full import *

ASR_MODEL = "facebook/mms-1b-all"
# MMS-ASR has script-specific Mandarin adapters
ASR_OVERRIDE = {"zho_Hans": "cmn-script_simplified", "zho_Hant": "cmn-script_traditional"}
SEM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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


def stratum(c):
    if c is None or c != c:
        return "unscorable"
    return "good" if c <= 0.4 else "medium" if c <= 0.7 else "poor"


def main():
    import numpy as np
    import torch
    import scipy.io.wavfile
    from collections import defaultdict
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    from sentence_transformers import SentenceTransformer

    device = "cuda:0"
    manifest = {r["item_id"]: r for r in iter_jsonl(DATA / "manifest.jsonl")}
    meta = [r for r in iter_jsonl(DATA / "tts_metadata.jsonl") if r.get("audio_path")]
    by_code = defaultdict(list)
    for r in meta:
        code = r.get("asr_lang") or r["tts_model"].split("mms-tts-")[-1]
        code = ASR_OVERRIDE.get(r["nllb_code"], code)
        by_code[code].append(r)

    processor = AutoProcessor.from_pretrained(ASR_MODEL)
    asr = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).to(device).eval()

    out = []
    for code, rows in sorted(by_code.items()):
        try:
            processor.tokenizer.set_target_lang(code)
            asr.load_adapter(code)
            ok = True
        except Exception as e:
            ok = False
            print(f"[{code}] no MMS-ASR adapter ({type(e).__name__}); {len(rows)} files unscorable", flush=True)
        for r in rows:
            m = manifest[r["item_id"]]
            base = {"item_id": r["item_id"], "item_type": r["item_type"], "root_id": r["root_id"],
                    "nllb_code": r["nllb_code"], "language": r["language"], "tts_model": r["tts_model"],
                    "tts_route": r["tts_route"], "asr_model": ASR_MODEL, "asr_lang": code,
                    "reference_text": m["text"][:TTS_CHAR_CAP], "recovered_transcript": None,
                    "wer": None, "cer": None, "quality_stratum": "unscorable",
                    "degenerate_audio": None, "tts_text_truncated": r["tts_text_truncated"]}
            if ok:
                sr, wav = scipy.io.wavfile.read(FULL / r["audio_path"])
                wav = wav.astype(np.float32)
                if wav.size and abs(wav).max() > 1.5:
                    wav /= 32768.0
                if sr != 16000:
                    from math import gcd
                    from scipy.signal import resample_poly
                    g = gcd(16000, sr)
                    wav, sr = resample_poly(wav, 16000 // g, sr // g).astype(np.float32), 16000
                inp = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                with torch.no_grad():
                    ids = torch.argmax(asr(**inp).logits, dim=-1)[0]
                rec = processor.decode(ids).strip()
                ref_n, rec_n = norm(base["reference_text"]), norm(rec)
                c, w = error_rate(ref_n, rec_n, "c"), error_rate(ref_n, rec_n, "w")
                base.update({"recovered_transcript": rec, "degenerate_audio": len(rec_n) == 0,
                             "wer": round(w, 4) if w == w else None, "cer": round(c, 4) if c == c else None,
                             "quality_stratum": stratum(c)})
            out.append(base)
        print(f"[{code}] {len(rows)} files", flush=True)

    sem = SentenceTransformer(SEM_MODEL, device=device)
    idx = [i for i, r in enumerate(out) if r["recovered_transcript"] is not None]
    a = sem.encode([out[i]["reference_text"] for i in idx], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    b = sem.encode([out[i]["recovered_transcript"] or " " for i in idx], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    for i, x, y in zip(idx, a, b):
        out[i]["semantic_similarity"] = round(float(np.dot(x, y)), 4)
    write_jsonl(DATA / "tts_quality.jsonl", out)

    from collections import Counter
    print(Counter(r["quality_stratum"] for r in out))


if __name__ == "__main__":
    main()
