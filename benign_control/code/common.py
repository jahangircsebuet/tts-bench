"""Shared loaders/helpers for the benign_control/ pipeline.

Reads from full/data only for cross-referencing (never writes there); all
outputs go directly under benign_control/results/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repository root
HERE = Path(__file__).resolve().parents[1]           # benign_control/
RESULTS = HERE / "results"
REPO = ROOT                                           # repository root (config/ lives here)

sys.path.insert(0, str(REPO))


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
