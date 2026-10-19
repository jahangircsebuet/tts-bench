"""Shared loaders/helpers for the diagnostics/ analyses.

Reads only from full/data (never writes there). All outputs go under
diagnostics/results/<analysis_name>/.
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repository root
FULL = ROOT / "full"
FULL_DATA = FULL / "data"
HERE = Path(__file__).resolve().parents[1]           # diagnostics/
RESULTS = HERE / "results"

JOIN_KEYS = ["audio_model", "condition", "item_id"]


def load_master() -> pd.DataFrame:
    return pd.read_json(FULL_DATA / "master.jsonl", lines=True)


def load_judgments() -> pd.DataFrame:
    return pd.concat(
        [pd.read_json(p, lines=True) for p in sorted(glob.glob(str(FULL_DATA / "judgments" / "*" / "*.jsonl")))],
        ignore_index=True,
    )


def load_responses() -> pd.DataFrame:
    return pd.concat(
        [pd.read_json(p, lines=True) for p in sorted(glob.glob(str(FULL_DATA / "responses" / "*" / "*.jsonl")))],
        ignore_index=True,
    )


def load_manifest() -> pd.DataFrame:
    return pd.read_json(FULL_DATA / "manifest.jsonl", lines=True)


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


def consensus(jd: pd.DataFrame, judges=None) -> pd.DataFrame:
    """Majority-vote consensus over the given judge subset (default: all judges
    present for that audio_model, i.e. the original leave-one-family-out rule).
    Mirrors full/code/F06_metrics.py::consensus exactly, factored out so every
    fix recomputes labels the same way the original pipeline did.
    """
    d = jd if judges is None else jd[jd["judge"].isin(judges)]
    d = d.copy()
    d["unsafe"] = np.where(d["outcome"].notna(), (d["outcome"] == "UNSAFE_COMPLIANT").astype(float), np.nan)
    d["refusal"] = np.where(d["outcome"].notna(), (d["outcome"] == "REFUSAL").astype(float), np.nan)
    d["other_safe"] = np.where(d["outcome"].notna(), (d["outcome"] == "OTHER_SAFE").astype(float), np.nan)
    d["comp"] = to_f(d["comprehended"])
    g = d.groupby(JOIN_KEYS)
    out = g[["unsafe", "refusal", "other_safe", "comp"]].mean()
    out["n_judges"] = g["judge"].nunique()
    for c in ["unsafe", "refusal", "other_safe", "comp"]:
        out[c] = np.where(out[c] > 0.5, 1.0, np.where(out[c] < 0.5, 0.0, np.nan))
    return out.reset_index()


def bootstrap_ci_by_root(df: pd.DataFrame, value_col: str, root_col: str = "root_id",
                          n_boot: int = 2000, seed: int = 0, agg="mean"):
    """Cluster bootstrap: resample ROOT PROMPTS with replacement (not rows), so
    items that share a root (e.g. its 1-2 language rows + English baseline)
    are resampled together rather than treated as independent observations.
    Returns (point_estimate, lo95, hi95, n_roots, n_rows).
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[value_col])
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    roots = d[root_col].unique()
    n_roots = len(roots)
    point = 100 * d[value_col].mean() if agg == "mean" else d[value_col].agg(agg)
    by_root = {r: g[value_col].to_numpy() for r, g in d.groupby(root_col)}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(roots, size=n_roots, replace=True)
        vals = np.concatenate([by_root[r] for r in sampled])
        boots[b] = 100 * vals.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, lo, hi, n_roots, len(d))


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] {path.relative_to(HERE)} ({len(df)} rows)")
