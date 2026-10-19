#!/usr/bin/env python3
"""Replace the original paper's raw-coefficient-magnitude ranking (Table
tab:logreg: "model identity is the strongest predictor... larger in
magnitude than any individual harm category") with (a) average marginal
effects (AME, which are comparable across dummy-coded features in a way raw
logit coefficients are not, since coefficient magnitude depends on the
reference category's baseline rate) and (b) likelihood-ratio tests comparing
nested models (full vs. full-minus-{model+modality} vs. full-minus-category)
to test whether model/modality actually explain more deviance than the
harm-category block, rather than inferring this from coefficient size alone.

Refits the ORIGINAL paper's exact feature set (one-hot model identity,
modality, TTS route, sample origin, category) on the `unsafe` label, via
statsmodels Logit with cluster-robust SEs by root_id (same root-awareness as
language_modality_interaction_model.py), then reports AME + LR tests.
CPU-only; no GPU needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_master, save

N_BOOT = 300

OUT = RESULTS / "average_marginal_effects"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_GROUPS = {
    "model_modality": ["audio_model", "modality"],
    "tts_route": ["lang_tts_route"],
    "sample_origin": ["sample_origin"],
    "category": ["category"],
}


def build_X(d: pd.DataFrame, exclude_groups=()):
    cols = []
    for g, feats in FEATURE_GROUPS.items():
        if g in exclude_groups:
            continue
        cols.extend(feats)
    X = pd.get_dummies(d[cols], columns=cols, drop_first=False).astype(float)
    X = sm.add_constant(X)
    keep = X.nunique() > 1
    keep["const"] = True
    return X.loc[:, keep]


def main():
    master = load_master()
    d = master[master["item_type"] == "lang"].dropna(subset=["unsafe", "category", "lang_tts_route", "sample_origin"]).copy()
    print(f"n = {len(d):,} lang-item rows")

    y = d["unsafe"].astype(int)
    X_full = build_X(d)
    full_model = sm.Logit(y, X_full)
    full_res = full_model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["root_id"].values}, maxiter=200)

    # ---- (a) Average Marginal Effects via root-level cluster bootstrap on a REGULARIZED fit ----
    # statsmodels' get_margeff() uses the asymptotic delta method on the unpenalized MLE's Jacobian,
    # which is numerically unusable here: `unsafe` is rare enough (same near-separation issue as
    # in language_modality_interaction_model.py) that the Hessian is near-singular and margeff SEs
    # explode to ~1e5-1e12 with p=1.0 (seen on
    # a first run of this script -- kept as a documented negative result in README, not silently
    # dropped). We instead compute AME the way it is defined -- average over the sample of
    # [P(y=1 | x_j=1, rest observed) - P(y=1 | x_j=0, rest observed)] -- using a root-level cluster
    # bootstrap of an L2-regularized logistic fit for a numerically stable CI.
    terms = [c for c in X_full.columns if c != "const"]
    roots = d["root_id"].unique()
    by_root = {r: idx for r, idx in d.groupby("root_id").groups.items()}
    rng = np.random.default_rng(0)
    boot_ame = {t: [] for t in terms}
    for b in range(N_BOOT):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        idx = np.concatenate([by_root[r].to_numpy() for r in sampled])
        Xb, yb = X_full.loc[idx].drop(columns=["const"]), y.loc[idx]
        if yb.nunique() < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        clf.fit(Xb, yb)
        # evaluate the marginal effect on the FULL (non-bootstrapped) design, varying only term j
        X_eval = X_full.drop(columns=["const"]).copy()
        base_p = clf.predict_proba(X_eval)[:, 1]
        for t in terms:
            x1, x0 = X_eval.copy(), X_eval.copy()
            x1[t], x0[t] = 1.0, 0.0
            p1 = clf.predict_proba(x1)[:, 1]
            p0 = clf.predict_proba(x0)[:, 1]
            boot_ame[t].append(float(np.mean(p1 - p0)))
        if (b + 1) % 50 == 0:
            print(f"  AME bootstrap {b + 1}/{N_BOOT}", flush=True)

    ame_rows = []
    for t in terms:
        arr = np.array(boot_ame[t])
        if len(arr) == 0:
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ame_rows.append({"term": t, "ame_median": round(float(np.median(arr)), 4),
                         "ame_ci_lo": round(float(lo), 4), "ame_ci_hi": round(float(hi), 4),
                         "n_boot_used": len(arr)})
    ame_df = pd.DataFrame(ame_rows).sort_values("ame_median", key=lambda s: s.abs(), ascending=False)
    save(ame_df, OUT / "average_marginal_effects.csv")

    # ---- (b) likelihood-ratio tests: full model vs. full-minus-{group} ----
    lr_rows = []
    for group in ["model_modality", "category", "tts_route", "sample_origin"]:
        X_reduced = build_X(d, exclude_groups=[group])
        reduced_res = sm.Logit(y, X_reduced).fit(disp=0, maxiter=200)  # plain MLE (LR test needs the MLE log-lik, not the sandwich-cov one; cluster-robust cov doesn't change the fitted loglik)
        lr_stat = 2 * (full_res.llf - reduced_res.llf)
        df_diff = len(full_res.params) - len(reduced_res.params)
        from scipy import stats as sstats
        p_lr = sstats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
        lr_rows.append({"dropped_group": group, "df_diff": df_diff, "llf_full": round(full_res.llf, 2),
                        "llf_reduced": round(reduced_res.llf, 2), "lr_statistic": round(lr_stat, 2),
                        "p_value": p_lr, "deviance_explained_by_group": round(lr_stat, 2)})
    lr_df = pd.DataFrame(lr_rows).sort_values("deviance_explained_by_group", ascending=False)
    save(lr_df, OUT / "likelihood_ratio_tests.csv")

    print("\n== likelihood-ratio deviance explained by each feature GROUP (higher = more explanatory power) ==")
    print(lr_df.to_string(index=False))
    print("\n== top 10 |AME| terms (comparable across dummy-coded features, unlike raw coefficients) ==")
    print(ame_df.head(10).to_string(index=False))

    # ---- verdict: does "model/modality > category" survive this more rigorous treatment? ----
    mm_dev = lr_df.loc[lr_df["dropped_group"] == "model_modality", "lr_statistic"].iloc[0]
    cat_dev = lr_df.loc[lr_df["dropped_group"] == "category", "lr_statistic"].iloc[0]
    verdict = ("CONFIRMED: model+modality explain MORE deviance than category"
               if mm_dev > cat_dev else
               "NOT CONFIRMED: category explains more deviance than model+modality under this test")
    print(f"\nVerdict (LR-test, replaces raw-coefficient-magnitude claim): {verdict} "
          f"(model+modality LR={mm_dev:.1f} vs. category LR={cat_dev:.1f}, {len(FEATURE_GROUPS['category'])} vs "
          f"{len(d['category'].unique())} category levels)")
    (OUT / "verdict.txt").write_text(
        f"model_modality LR statistic: {mm_dev:.2f}\ncategory LR statistic: {cat_dev:.2f}\n{verdict}\n")


if __name__ == "__main__":
    main()
