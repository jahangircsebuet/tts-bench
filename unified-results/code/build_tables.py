#!/usr/bin/env python3
"""Build every unified, 6-model results table from the normalized master
frames (load_unified.py). One script, one source of truth, so every number
in the paper's unified tables traces back to here.

NOTE on scope: this unifies every table that can be rebuilt from existing
data (reusing the identical synthesized audio and the existing judgments).
Four diagnostic checks remain 2-model-only because they would require NEW
judge/reclassifier runs to extend, not just re-aggregation, and are reported
as such rather than silently extended: judge-reliability full pairwise
agreement (needs a 3rd judge run on 4 more models), the rubric-sensitivity
re-annotation, the OTHER_SAFE 7-way reclassification, and the documented-
language-support audit (needs new vendor-documentation research for 4 more
models). See the paper's Limitations section.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_unified import (MODEL_ORDER, MODEL_ORDER_WITH_VARIANTS, MODEL_META, load_unsafe_master,
                          load_benign_master, wilson, rate, root_bootstrap_ci)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def save(df, name):
    df.to_csv(RESULTS / name, index=False)
    print(f"[OK] {name} ({len(df)} rows)")


def order_models(df, col="model", with_variants=False):
    order = MODEL_ORDER_WITH_VARIANTS if with_variants else MODEL_ORDER
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def canonical_only(df, col="model"):
    """Drop judge-config variant pseudo-models (e.g. ultravox_primary2judge) --
    use for tables that aren't about judge-configuration itself."""
    return df[df[col].isin(MODEL_ORDER)].copy()


# ---------------------------------------------------------------- T1: overall
def build_overall(unsafe):
    rows = []
    for (model, cond), g in unsafe.groupby(["model", "condition"]):
        u = g["unsafe"].dropna()
        lo, hi = wilson(u.sum(), len(u))
        cr_lo, cr_hi = root_bootstrap_ci(g, "comp", n_boot=500)
        rows.append({
            "model": model, "judge_config": "+".join(MODEL_META[model]["judges"]),
            "condition": cond, "n": len(g), "n_scored": len(u),
            "n_disputed": int(g["unsafe"].isna().sum()),
            "UCR_pct": round(rate(g["unsafe"]), 2), "UCR_lo": round(lo, 2), "UCR_hi": round(hi, 2),
            "RR_pct": round(rate(g["refusal"]), 2),
            "CR_pct": round(rate(g["comp"]), 2), "CR_lo": round(cr_lo, 2), "CR_hi": round(cr_hi, 2),
        })
    return order_models(pd.DataFrame(rows), with_variants=True)


# ---------------------------------------------------------------- T2: 4-way
def build_fourway(unsafe):
    rows = []
    for (model, cond), g in unsafe.groupby(["model", "condition"]):
        c, u = g["comp"], g["unsafe"]
        both = pd.concat([c, u], axis=1).dropna()
        n = len(both)
        if n == 0:
            continue
        aligned = ((both["comp"] == 1) & (both["unsafe"] == 0)).mean() * 100
        align_fail = ((both["comp"] == 1) & (both["unsafe"] == 1)).mean() * 100
        faos = ((both["comp"] == 0) & (both["unsafe"] == 0)).mean() * 100
        comp_fail_unsafe = ((both["comp"] == 0) & (both["unsafe"] == 1)).mean() * 100
        rows.append({"model": model, "condition": cond, "n_scored": n,
                     "genuinely_aligned_pct": round(aligned, 2), "alignment_failure_pct": round(align_fail, 2),
                     "false_appearance_safety_pct": round(faos, 2), "comp_failure_unsafe_pct": round(comp_fail_unsafe, 2)})
    return order_models(pd.DataFrame(rows), with_variants=True)


# ---------------------------------------------------------------- T3: resource tier
def build_tier(unsafe, conditions=("speech_lang",)):
    rows = []
    sub = unsafe[unsafe["condition"].isin(conditions)]
    for (model, cond, tier), g in sub.groupby(["model", "condition", "resource_tier"]):
        u = g["unsafe"].dropna()
        rows.append({"model": model, "condition": cond, "resource_tier": tier, "n": len(g),
                     "UCR_pct": round(rate(g["unsafe"]), 2), "CR_pct": round(rate(g["comp"]), 2)})
    return order_models(pd.DataFrame(rows))


# ---------------------------------------------------------------- T4: benign
def build_benign(benign):
    benign = benign.copy()
    benign["item_type"] = np.where(benign["condition"].str.endswith("en"), "en", "lang")
    rows = []
    for (model, cond), g in benign.groupby(["model", "condition"]):
        if cond.endswith("en"):
            rows.append({"model": model, "condition": cond, "resource_tier": "High", "n": len(g),
                         "benign_CR_pct": round(rate(g["comp"]), 2),
                         "inappropriate_refusal_pct": round(rate(g["inappropriate_refusal"]), 2)})
        else:
            for tier, gt in g.groupby("resource_tier"):
                rows.append({"model": model, "condition": cond, "resource_tier": tier, "n": len(gt),
                             "benign_CR_pct": round(rate(gt["comp"]), 2),
                             "inappropriate_refusal_pct": round(rate(gt["inappropriate_refusal"]), 2)})
    return order_models(pd.DataFrame(rows))


# ---------------------------------------------------------------- T5: TTS route stratum
def build_route(unsafe, cond="speech_lang"):
    sub = unsafe[(unsafe["condition"] == cond) & unsafe["tts_route"].notna()].copy()
    stratum_map = {"mms": "native", "piper": "native", "mms_alias": "alias", "piper_alias": "proxy"}
    sub["stratum"] = sub["tts_route"].map(stratum_map)
    rows = []
    for (model, stratum), g in sub.groupby(["model", "stratum"]):
        rows.append({"model": model, "stratum": stratum, "n": len(g),
                     "UCR_pct": round(rate(g["unsafe"]), 2), "CR_pct": round(rate(g["comp"]), 2)})
    return order_models(pd.DataFrame(rows))


# ---------------------------------------------------------------- T6: RSC
def build_rsc(unsafe):
    rows = []
    for model, gm in unsafe.groupby("model"):
        for modality, gmod in gm.groupby("modality"):
            per_root = gmod.dropna(subset=["unsafe"]).groupby("root_id")["unsafe"].nunique()
            per_root = per_root[gmod.dropna(subset=["unsafe"]).groupby("root_id").size() >= 2]
            if len(per_root) == 0:
                continue
            rsc = (per_root == 1).mean() * 100
            rows.append({"model": model, "modality": modality, "n_roots": len(per_root), "RSC_pct": round(rsc, 2)})
    return order_models(pd.DataFrame(rows))


# ---------------------------------------------------------------- T7: sample origin (MSG/SAG)
def build_origin_gaps(unsafe):
    """MSG (text->speech UCR gap) and SAG (vs English's own MSG) by sample_origin,
    per model. Mirrors full/code/F06_metrics.py's T2c EXACTLY: a 'lang' item_id is a
    specific (root, language) pair -- NOT the same as root_id, since one root can be
    sampled into the 10,000-row lang manifest under several different languages, each
    with its OWN sample_origin tag (an earlier version of this function pivoted by
    root_id alone and silently collapsed/mis-tagged these, undercounting old_reuse by
    ~3x -- caught by cross-checking n against the already-published tab:origin before
    trusting this output). The correct unit is item_id (pivot text_lang vs speech_lang
    for the SAME lang item), with the English baseline joined in by root_id."""
    lang = unsafe[unsafe["item_type"] == "lang"][["model", "item_id", "root_id", "sample_origin"]].drop_duplicates()
    piv_lang = unsafe[unsafe["item_type"] == "lang"].pivot_table(
        index=["model", "item_id"], columns="condition", values="unsafe", aggfunc="first")
    W = lang.set_index(["model", "item_id"]).join(piv_lang, how="left").reset_index()

    en = unsafe[unsafe["item_type"] == "en"]
    en_piv = en.pivot_table(index=["model", "root_id"], columns="condition", values="unsafe", aggfunc="first")
    en_piv.columns = [f"en_{c}" for c in en_piv.columns]
    W = W.merge(en_piv.reset_index(), on=["model", "root_id"], how="left")

    rows = []
    for (model, origin), g in W.groupby(["model", "sample_origin"]):
        gs = g.dropna(subset=["text_lang", "speech_lang"])
        if len(gs) == 0:
            continue
        msg = 100 * (gs["speech_lang"].mean() - gs["text_lang"].mean())
        msg_en = 100 * (gs["en_speech_en"].mean() - gs["en_text_en"].mean())
        sag = msg - msg_en
        rows.append({"model": model, "sample_origin": origin, "n_both": len(gs),
                     "MSG": round(msg, 2), "SAG": round(sag, 2)})
    return order_models(pd.DataFrame(rows))


# ---------------------------------------------------------------- T8: quality/truncation
def build_quality_trunc(unsafe, cond="speech_lang"):
    sub = unsafe[unsafe["condition"] == cond]
    rows = []
    def _to_bool(v):
        return np.nan if v is None or v != v else bool(v)

    for model, g in sub.groupby("model"):
        g = g.copy()
        g["audio_truncated"] = g["audio_truncated"].apply(_to_bool)
        tr = g.groupby("audio_truncated")["comp"].apply(rate)
        q = g.groupby("quality_stratum")["comp"].apply(rate) if g["quality_stratum"].notna().any() else pd.Series(dtype=float)
        rows.append({
            "model": model,
            "CR_truncated": round(tr.get(True, np.nan), 2), "CR_not_truncated": round(tr.get(False, np.nan), 2),
            "CR_good": round(q.get("good", np.nan), 2), "CR_medium": round(q.get("medium", np.nan), 2),
            "CR_poor": round(q.get("poor", np.nan), 2), "CR_unscorable": round(q.get("unscorable", np.nan), 2),
        })
    return order_models(pd.DataFrame(rows))


def main():
    print("Loading unified master frames...")
    unsafe = load_unsafe_master()
    benign = load_benign_master()
    print(f"unsafe: {len(unsafe):,} rows, benign: {len(benign):,} rows, 6 models confirmed in each")

    unsafe6 = canonical_only(unsafe)  # drop the ultravox_primary2judge variant for non-judge-config tables

    save(build_overall(unsafe), "T1_overall_6models.csv")
    save(build_fourway(unsafe), "T2_fourway_6models.csv")
    save(build_tier(unsafe6), "T3_tier_speech_lang_6models.csv")
    save(build_tier(unsafe6, conditions=("text_lang", "speech_lang", "text_en", "speech_en")), "T3b_tier_all_conditions_6models.csv")
    save(build_benign(benign), "T4_benign_6models.csv")
    save(build_route(unsafe6), "T5_route_6models.csv")
    save(build_rsc(unsafe6), "T6_rsc_6models.csv")
    save(build_origin_gaps(unsafe6), "T7_origin_gaps_6models.csv")
    save(build_quality_trunc(unsafe6), "T8_quality_trunc_6models.csv")

    print("\n== T1 overall (speech_lang only) ==")
    t1 = pd.read_csv(RESULTS / "T1_overall_6models.csv")
    print(t1[t1["condition"] == "speech_lang"].to_string(index=False))


if __name__ == "__main__":
    main()
