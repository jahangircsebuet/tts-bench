#!/usr/bin/env python3
"""Root-prompt cluster bootstrap 95% CIs for CR, the 4-way comprehension x
safety breakdown (genuinely aligned / false appearance of safety / etc.), and
the aggregate gap metrics (CLSG, MSG, SAG). Also flags per-language cells with
small speech-mode n so readers don't over-read noisy per-language numbers.

Resampling is done by ROOT PROMPT (not row), because a root's English-baseline
row and its language row(s) are not independent draws -- they share the same
underlying request. 2000 bootstrap replicates. CPU-only.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, bootstrap_ci_by_root, load_master, rate, save

OUT = RESULTS / "bootstrap_confidence_intervals"
N_BOOT = 2000
SMALL_N_THRESHOLD = 100  # flag per-language speech cells below this n


def ci_table_overall(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (am, cond), g in master.groupby(["audio_model", "condition"]):
        pt, lo, hi, n_roots, n_rows = bootstrap_ci_by_root(g, "comp", n_boot=N_BOOT)
        rows.append({"audio_model": am, "condition": cond, "metric": "CR_pct",
                      "point": round(pt, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                      "n_roots": n_roots, "n_rows": n_rows})
    return pd.DataFrame(rows)


def ci_table_fourway(master: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap CI for each of the 4 comprehension x safety interpretation
    shares, per model/condition (focused on speech_lang, the condition
    combining both perturbations, but computed for all conditions)."""
    lab = {(1, 0): "Genuinely_aligned", (1, 1): "Alignment_failure",
           (0, 0): "False_appearance_of_safety", (0, 1): "Comprehension_failure_unsafe"}
    rows = []
    for (am, cond), g in master.groupby(["audio_model", "condition"]):
        g = g.dropna(subset=["unsafe", "comp"]).copy()
        for (c, u), name in lab.items():
            g["_hit"] = ((g["comp"] == c) & (g["unsafe"] == u)).astype(float)
            pt, lo, hi, n_roots, n_rows = bootstrap_ci_by_root(g, "_hit", n_boot=N_BOOT)
            rows.append({"audio_model": am, "condition": cond, "interpretation": name,
                         "point_pct": round(pt, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                         "n_roots": n_roots, "n_rows": n_rows})
    return pd.DataFrame(rows)


def ci_table_gaps(master: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap CI for MSG and CLSG (ASR_speech - ASR_text, and ASR_lang -
    ASR_English), by resampling roots and recomputing the gap each time so the
    CI accounts for the fact that text/speech/English values for the same root
    are correlated, not independent."""
    lang_items = master[master["item_type"] == "lang"][
        ["audio_model", "item_id", "root_id", "language", "nllb_code"]].drop_duplicates()
    W = lang_items.set_index(["audio_model", "item_id"])

    def piv(df, cond, tag):
        d = df[df["condition"] == cond]
        return d.set_index(["audio_model", "item_id"])[["unsafe"]].rename(columns={"unsafe": f"{tag}_unsafe"})

    for cond, tag in [("text_lang", "t"), ("speech_lang", "s")]:
        W = W.join(piv(master, cond, tag), how="left")
    en = master[master["item_type"] == "en"]
    en_t = en[en["condition"] == "text_en"].set_index(["audio_model", "root_id"])[["unsafe"]].rename(
        columns={"unsafe": "en_t_unsafe"})
    en_s = en[en["condition"] == "speech_en"].set_index(["audio_model", "root_id"])[["unsafe"]].rename(
        columns={"unsafe": "en_s_unsafe"})
    W = W.reset_index().merge(en_t.reset_index(), on=["audio_model", "root_id"], how="left") \
                       .merge(en_s.reset_index(), on=["audio_model", "root_id"], how="left")

    rng = np.random.default_rng(0)
    rows = []
    for am, g in W.groupby("audio_model"):
        gs = g[g["s_unsafe"].notna() & g["t_unsafe"].notna()]
        gt = g[g["t_unsafe"].notna()]

        def msg_point(df):
            return rate(df["s_unsafe"]) - rate(df["t_unsafe"])

        def sag_point(df):
            return (rate(df["s_unsafe"]) - rate(df["t_unsafe"])) - (rate(df["en_s_unsafe"]) - rate(df["en_t_unsafe"]))

        def clsg_point(df):
            return rate(df["t_unsafe"]) - rate(df["en_t_unsafe"])

        roots_both = gs["root_id"].unique()
        by_root_both = {r: idx for r, idx in gs.groupby("root_id").groups.items()}
        msg_boot, sag_boot = [], []
        for _ in range(N_BOOT):
            sampled = rng.choice(roots_both, size=len(roots_both), replace=True)
            idx = np.concatenate([by_root_both[r].to_numpy() for r in sampled])
            sub = gs.loc[idx]
            msg_boot.append(msg_point(sub))
            sag_boot.append(sag_point(sub))
        msg_lo, msg_hi = np.percentile(msg_boot, [2.5, 97.5])
        sag_lo, sag_hi = np.percentile(sag_boot, [2.5, 97.5])

        roots_text = gt["root_id"].unique()
        by_root_text = {r: idx for r, idx in gt.groupby("root_id").groups.items()}
        clsg_boot = []
        for _ in range(N_BOOT):
            sampled = rng.choice(roots_text, size=len(roots_text), replace=True)
            idx = np.concatenate([by_root_text[r].to_numpy() for r in sampled])
            clsg_boot.append(clsg_point(gt.loc[idx]))
        clsg_lo, clsg_hi = np.percentile(clsg_boot, [2.5, 97.5])

        rows.append({
            "audio_model": am, "n_roots_both": len(roots_both), "n_roots_text": len(roots_text),
            "MSG": round(msg_point(gs), 3), "MSG_ci_lo": round(msg_lo, 3), "MSG_ci_hi": round(msg_hi, 3),
            "SAG": round(sag_point(gs), 3), "SAG_ci_lo": round(sag_lo, 3), "SAG_ci_hi": round(sag_hi, 3),
            "CLSG_text": round(clsg_point(gt), 3), "CLSG_ci_lo": round(clsg_lo, 3), "CLSG_ci_hi": round(clsg_hi, 3),
        })
    return pd.DataFrame(rows)


def per_language_small_n_flags(master: pd.DataFrame) -> pd.DataFrame:
    """Per-language speech-mode CR with bootstrap CI and an explicit small-n flag."""
    sp = master[(master["item_type"] == "lang") & (master["condition"] == "speech_lang")]
    rows = []
    for (am, lang), g in sp.groupby(["audio_model", "language"]):
        pt, lo, hi, n_roots, n_rows = bootstrap_ci_by_root(g, "comp", n_boot=N_BOOT)
        rows.append({
            "audio_model": am, "language": lang, "nllb_code": g["nllb_code"].iloc[0],
            "n_speech": len(g), "n_roots": n_roots,
            "CR_speech_pct": round(pt, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
            "ci_width": round(hi - lo, 2),
            "small_n_flag": bool(len(g) < SMALL_N_THRESHOLD),
        })
    df = pd.DataFrame(rows).sort_values(["audio_model", "ci_width"], ascending=[True, False])
    return df


def main():
    master = load_master()

    save(ci_table_overall(master), OUT / "CR_overall_ci.csv")
    save(ci_table_fourway(master), OUT / "fourway_interpretation_ci.csv")
    save(ci_table_gaps(master), OUT / "gaps_MSG_CLSG_SAG_ci.csv")
    per_lang = per_language_small_n_flags(master)
    save(per_lang, OUT / "per_language_CR_ci_with_small_n_flags.csv")

    n_flagged = int(per_lang["small_n_flag"].sum())
    print(f"\n{n_flagged}/{len(per_lang)} (model, language) speech-mode cells have n < {SMALL_N_THRESHOLD}"
          f" and are flagged small_n_flag=True.")
    print("\nWidest-CI (least certain) per-language cells:")
    print(per_lang.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
