"""Shared loaders/helpers for the robustness/ analyses.

Reads from full/data (never writes there) and, for the resource-tier
lookup, from the repo root's config/ (the same place the original pipeline
itself reads from -- see full/code/common_full.py's own REPO/config/...
paths for precedent).
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repository root
FULL = ROOT / "full"
FULL_DATA = FULL / "data"
HERE = Path(__file__).resolve().parents[1]           # robustness/
RESULTS = HERE / "results"
REPO = ROOT                                           # repository root (config/ lives here)

sys.path.insert(0, str(REPO))


def load_master() -> pd.DataFrame:
    return pd.read_json(FULL_DATA / "master.jsonl", lines=True)


def load_manifest() -> pd.DataFrame:
    return pd.read_json(FULL_DATA / "manifest.jsonl", lines=True)


def load_judgments() -> pd.DataFrame:
    return pd.concat(
        [pd.read_json(p, lines=True) for p in sorted(glob.glob(str(FULL_DATA / "judgments" / "*" / "*.jsonl")))],
        ignore_index=True,
    )


def load_tts_quality() -> pd.DataFrame:
    return pd.read_json(FULL_DATA / "tts_quality.jsonl", lines=True)


def load_responses() -> pd.DataFrame:
    return pd.concat(
        [pd.read_json(p, lines=True) for p in sorted(glob.glob(str(FULL_DATA / "responses" / "*" / "*.jsonl")))],
        ignore_index=True,
    )


def load_resource_tier_map() -> dict:
    """nllb_code -> 'High'/'Low', from config/language_metadata.py's
    resource_level_nllb field (the NLLB paper's own binary resource
    categorization; this dict is already used elsewhere in the repo, e.g.
    src/11_result-analysis). Covers all 100 benchmark languages."""
    from config.language_metadata import LANG_NAME_TO_CODE
    out = {}
    for _, v in dict.items(LANG_NAME_TO_CODE):
        out[v["nllb_code"]] = v.get("resource_level_nllb")
    return out


def to_f(s: pd.Series) -> pd.Series:
    return s.astype(object).map(lambda v: np.nan if v is None or v != v else float(bool(v))).astype(float)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def rate(s: pd.Series) -> float:
    s = s.dropna()
    return 100 * s.mean() if len(s) else np.nan


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum(((a == v).mean() * (b == v).mean()) for v in np.union1d(a, b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] {path.relative_to(HERE)} ({len(df)} rows)")


def patch_numpy_for_fasttext():
    """fasttext's predict() calls np.array(obj, copy=False) which raises
    under numpy>=2.0 when a real copy is needed. Patch np.array locally
    (not globally destructive -- restores copy=False to np.asarray
    semantics only) so fasttext/GlotLID can be used as-is."""
    _orig = np.array

    def patched(obj, *a, copy=None, **kw):
        if copy is False:
            return np.asarray(obj)
        return _orig(obj, *a, **({} if copy is None else {"copy": copy}), **kw)

    np.array = patched


def load_glotlid():
    """Load the cached GlotLID fastText LID model (cis-lmu/glotlid)."""
    patch_numpy_for_fasttext()
    import fasttext
    path = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/*/model.bin"))[0]
    return fasttext.load_model(path)
