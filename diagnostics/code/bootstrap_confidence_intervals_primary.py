#!/usr/bin/env python3
"""Addendum to bootstrap_confidence_intervals.py -- root-level bootstrap 95%
CIs for CR and UCR under the PRIMARY (common-judge, Qwen3-14B+Aya-8B)
consensus specifically, since that consensus definition changed the primary
cross-model comparison after the original bootstrap was run against the
master.jsonl leave-one-family-out consensus. CPU-only.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, bootstrap_ci_by_root, load_master, save
from judge_consensus_primary_sensitivity import common_consensus

OUT = RESULTS / "bootstrap_confidence_intervals"
N_BOOT = 2000


def main():
    master = load_master()
    d = common_consensus(master)

    import pandas as pd
    rows = []
    for (am, cond), g in d.groupby(["audio_model", "condition"]):
        pt_cr, lo_cr, hi_cr, n_roots, n_rows = bootstrap_ci_by_root(g, "comp_common", n_boot=N_BOOT)
        pt_u, lo_u, hi_u, _, _ = bootstrap_ci_by_root(g, "unsafe_common", n_boot=N_BOOT)
        rows.append({
            "audio_model": am, "condition": cond,
            "CR_primary": round(pt_cr, 2), "CR_primary_ci_lo": round(lo_cr, 2), "CR_primary_ci_hi": round(hi_cr, 2),
            "UCR_primary": round(pt_u, 3), "UCR_primary_ci_lo": round(lo_u, 3), "UCR_primary_ci_hi": round(hi_u, 3),
            "n_roots": n_roots, "n_rows": n_rows,
        })
    df = pd.DataFrame(rows)
    save(df, OUT / "CR_UCR_overall_ci_PRIMARY_common_judge.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
