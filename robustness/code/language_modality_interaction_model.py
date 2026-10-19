#!/usr/bin/env python3
"""Formally test the language x modality interaction instead of just
describing the combined effect as "roughly additive".

Fits outcome ~ non_english + speech + non_english:speech + resource_tier +
tts_stratum + category, for outcome in {comprehended, unsafe}, via logistic
regression with CLUSTER-ROBUST standard errors grouped by root_id (not a
full random-intercept mixed model: with ~5,000+ distinct root_id values, a
true mixed-effects fit is both numerically fragile and slow; cluster-robust
("root-aware") sandwich SEs are the standard, tractable way to account for
the fact that a root's English/speech/text rows are not independent draws).

Fit once pooled (both audio models, with audio_model as a covariate) and
once per audio_model, since the paper's own findings suggest the two models
may not share the same interaction structure.

CPU-only (statsmodels); no GPU needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_master, load_resource_tier_map, save

COEF_SANITY_THRESHOLD = 15.0  # |coef| beyond this on a logit scale (OR > e^15) signals
                               # quasi-complete separation, not a real effect size
N_BOOT_UNSTABLE = 500

OUT = RESULTS / "language_modality_interaction_model"
OUT.mkdir(parents=True, exist_ok=True)

STRATUM = {"mms": "native", "piper": "native", "mms_alias": "alias", "piper_alias": "proxy"}


def build_design(master: pd.DataFrame) -> pd.DataFrame:
    d = master.copy()
    d["non_english"] = (d["condition"].isin(["text_lang", "speech_lang"])).astype(int)
    d["speech"] = (d["condition"].isin(["speech_en", "speech_lang"])).astype(int)
    d["interaction"] = d["non_english"] * d["speech"]

    tier_map = load_resource_tier_map()
    d["resource_tier"] = d["nllb_code"].map(tier_map)
    d.loc[d["item_type"] == "en", "resource_tier"] = "High"  # English baseline

    d["tts_stratum"] = d["tts_route"].map(STRATUM).fillna("text_only")
    d.loc[d["modality"] == "text", "tts_stratum"] = "text_only"

    return d.dropna(subset=["comp", "unsafe", "resource_tier", "category"])


def build_X(d: pd.DataFrame, include_model: bool):
    X = pd.get_dummies(
        d[["non_english", "speech", "interaction", "resource_tier", "tts_stratum", "category"]
          + (["audio_model"] if include_model else [])],
        columns=["resource_tier", "tts_stratum", "category"] + (["audio_model"] if include_model else []),
        drop_first=True,
    ).astype(float)
    X = sm.add_constant(X)
    keep = X.nunique() > 1
    keep["const"] = True  # always keep the intercept even though its nunique()==1 by construction
    return X.loc[:, keep]


def bootstrap_penalized_ci(d: pd.DataFrame, outcome: str, include_model: bool, term: str, n_boot=N_BOOT_UNSTABLE, seed=0):
    """Root-level cluster bootstrap CI for one term's coefficient under an
    L2-penalized (ridge) logistic fit, used as the stable fallback when the
    unpenalized MLE shows quasi-complete separation."""
    rng = np.random.default_rng(seed)
    roots = d["root_id"].unique()
    by_root = {r: idx for r, idx in d.groupby("root_id").groups.items()}
    coefs = []
    for _ in range(n_boot):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        idx = np.concatenate([by_root[r].to_numpy() for r in sampled])
        sub = d.loc[idx]
        Xb = build_X(sub, include_model)
        yb = sub[outcome].astype(int)
        if term not in Xb.columns or yb.nunique() < 2:
            continue
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=500, solver="lbfgs")
        clf.fit(Xb.drop(columns=["const"]), yb)
        j = list(Xb.drop(columns=["const"]).columns).index(term)
        coefs.append(clf.coef_[0][j])
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    return float(np.median(coefs)), float(lo), float(hi), len(coefs)


def fit_cluster_robust_logit(d: pd.DataFrame, outcome: str, include_model: bool, label: str):
    y = d[outcome].astype(int)
    X = build_X(d, include_model)
    model = sm.Logit(y, X)
    try:
        res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["root_id"].values}, maxiter=200)
        converged = res.mle_retvals.get("converged", True)
    except Exception as e:
        print(f"[warn] {label}/{outcome}: unpenalized fit raised {type(e).__name__}: {e}")
        res, converged = None, False

    rows = []
    for name in ["non_english", "speech", "interaction"]:
        if res is not None and name in res.params.index:
            coef = res.params[name]
            lo, hi = res.conf_int().loc[name]
            unstable = (not converged) or abs(coef) > COEF_SANITY_THRESHOLD or abs(hi - lo) > 2 * COEF_SANITY_THRESHOLD
        else:
            coef, lo, hi, unstable = np.nan, np.nan, np.nan, True

        row = {"label": label, "outcome": outcome, "term": name,
               "mle_coef": round(coef, 4) if coef == coef else None,
               "mle_ci_lo": round(lo, 4) if lo == lo else None,
               "mle_ci_hi": round(hi, 4) if hi == hi else None,
               "mle_p_value": round(res.pvalues[name], 5) if (res is not None and name in res.pvalues.index) else None,
               "mle_unstable_quasi_separation": bool(unstable)}
        if unstable:
            med, blo, bhi, n_eff = bootstrap_penalized_ci(d, outcome, include_model, name)
            row.update({"ridge_boot_median_coef": round(med, 4), "ridge_boot_ci_lo": round(blo, 4),
                        "ridge_boot_ci_hi": round(bhi, 4), "ridge_boot_n_eff": n_eff})
            print(f"[flag] {label}/{outcome}/{name}: unpenalized MLE unstable (quasi-complete "
                  f"separation, too few positive outcomes for stable estimation) -- reporting "
                  f"root-bootstrap L2-penalized fallback instead: coef~{med:.3f} [{blo:.3f}, {bhi:.3f}]")
        rows.append(row)
    terms_df = pd.DataFrame(rows)
    return res, terms_df


def main():
    master = load_master()
    d = build_design(master)
    print(f"design matrix: {len(d):,} rows (dropped {len(master) - len(d):,} with missing comp/unsafe/tier)")

    all_terms = []
    full_results = {}

    for outcome in ["comp", "unsafe"]:
        res, terms = fit_cluster_robust_logit(d, outcome, include_model=True, label="pooled_both_models")
        all_terms.append(terms)
        full_results[f"pooled_{outcome}"] = res
        for am, g in d.groupby("audio_model"):
            res_m, terms_m = fit_cluster_robust_logit(g, outcome, include_model=False, label=f"{am}_only")
            all_terms.append(terms_m)
            full_results[f"{am}_{outcome}"] = res_m

    terms_df = pd.concat(all_terms, ignore_index=True)
    save(terms_df, OUT / "interaction_terms.csv")

    # full coefficient tables for the pooled models, for transparency/reproducibility
    for key in ["pooled_comp", "pooled_unsafe"]:
        res = full_results[key]
        if res is None:
            continue
        full_tbl = pd.DataFrame({"term": res.params.index, "coef": res.params.values,
                                  "se_cluster_robust": res.bse.values, "p_value": res.pvalues.values})
        save(full_tbl, OUT / f"full_coefficients_{key}.csv")

    pd.set_option("display.width", 160)
    print("\n== interaction-term summary (non_english x speech) ==")
    print(terms_df[terms_df["term"] == "interaction"].to_string(index=False))
    print("\n== main effects ==")
    print(terms_df[terms_df["term"] != "interaction"].to_string(index=False))
    print(f"\nUnstable (quasi-separated) MLE cells flagged: {int(terms_df['mle_unstable_quasi_separation'].sum())}/{len(terms_df)}")


if __name__ == "__main__":
    main()
