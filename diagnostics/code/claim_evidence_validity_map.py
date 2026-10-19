#!/usr/bin/env python3
"""A compact claim-to-evidence validity map.

This is an editorial synthesis task, not a new numerical computation --
every cell below cites a specific table/section already in the paper (or
elsewhere in diagnostics/) rather than introducing a new number. The
script's job is to assemble it in one place, as structured data, so it can
be regenerated/audited rather than hand-typed into LaTeX with no record of
how it was derived, and to sanity-check that every "primary table" cell it
references actually exists in paper.tex.

CPU-only; no GPU needed. Output is both a CSV (for inspection) and a ready-
to-paste LaTeX table (inserted into paper.tex by hand afterward, as with the
other analyses in this folder).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS

PAPER = Path(__file__).resolve().parents[3] / "writeup" / "paper.tex"
OUT = RESULTS / "claim_evidence_validity_map"
OUT.mkdir(parents=True, exist_ok=True)

# Each row: (claim, dataset_subset, judge_config, tts_stratum, ci_method, remaining_caveat, evidence_table_label)
CLAIMS = [
    ("Comprehension collapses outside English/text (CR 99%->58%/53%)",
     "Paired text/speech items, both models", "Primary (2-judge common); holds under sensitivity too",
     "Pooled (native+alias+proxy)", "Root-level bootstrap", "Pooling strata partly masks a stratum-specific effect (see row below)",
     "tab:overall"),
    ("42-47% of non-English speech responses are false-appearance-of-safety",
     "speech\\_lang paired items", "Sensitivity (leave-one-family-out) ensemble",
     "Pooled", "Root-level bootstrap", "OTHER\\_SAFE/comprehended boundary uses the original one-line rubric, not the refined one (Section~\\ref{sec:method:judges}; quantified in rubric_subset_reannotation_2400.py below)",
     "tab:fourway"),
    ("Cross-model UCR gap nearly vanishes under primary judges; CR gap does not",
     "Paired text/speech items, both models", "Primary vs.\\ sensitivity, both reported",
     "Pooled", "Wilson (UCR), bootstrap (CR)", "Mechanism is unanimity-vs-majority consensus rule, not merely judge count",
     "tab:overall, tab:overall-sensitivity"),
    ("Native-voice CR differs sharply from alias/proxy-voice CR (Ultravox: 48% vs.\\ 83%)",
     "speech\\_lang items only", "Sensitivity ensemble consensus",
     "Explicitly stratified: native vs.\\ alias vs.\\ proxy", "None yet (point estimates, $n$=285-5,409 per stratum)",
     "No bootstrap CI computed per stratum; smallest strata ($n$=285) are the least certain",
     "tab:route"),
    ("Documented language support predicts comprehension (+15 to +28 points)",
     "speech\\_lang items", "Sensitivity ensemble consensus",
     "Pooled across strata", "None yet (point estimates)",
     "Gemma-3n's ``undocumented'' group is a documentation gap, not a disproof of support (Limitations)",
     "tab:langsupport"),
    ("OTHER\\_SAFE is dominated by language confusion, not generic off-topic-ness",
     "All unique OTHER\\_SAFE responses (23,529)", "Single reclassifier (Qwen3-14B), NOT the 3-judge ensemble",
     "Pooled", "None (counts only)", "Reclassifier is itself one more LLM judge, not human-validated (Limitations)",
     "tab:othersafe"),
    ("Model identity and modality outrank harm category as predictors of the unsafe label",
     "All lang-item rows, both models", "Sensitivity ensemble consensus (the `unsafe` field)",
     "TTS route included as a feature, not stratified", "None (point coefficients, no bootstrap SE)",
     "Cross-model comparison is observational (Section~\\ref{sec:method:metrics}), not paired by design",
     "tab:logreg"),
    ("Pairwise judge agreement is low (Cohen's $\\kappa$ 0.011-0.525)",
     "All judged items, both models", "All 3 judges pairwise (this is what is being measured)",
     "Pooled", "None (point $\\kappa$, no bootstrap CI)", "Single point estimate per judge pair; not re-derived on a bootstrap sample",
     "tab:judgeagree"),
]


def main():
    df = pd.DataFrame(CLAIMS, columns=["claim", "dataset_subset", "judge_config", "tts_stratum",
                                        "ci_method", "remaining_caveat", "evidence_label"])
    df.to_csv(OUT / "validity_map.csv", index=False)
    print(f"[OK] results/{OUT.name}/validity_map.csv ({len(df)} rows)")

    text = PAPER.read_text()
    existing_labels = set(re.findall(r"\\label\{([^}]*)\}", text))
    missing = []
    for row in CLAIMS:
        for lbl in row[-1].split(", "):
            if lbl not in existing_labels:
                missing.append(lbl)
    if missing:
        print(f"[WARN] evidence labels referenced that do not exist in paper.tex: {missing}")
    else:
        print("[OK] every evidence_label cited in the validity map exists as a \\label in paper.tex")

    # ---- emit ready-to-paste LaTeX table ----
    lines = [r"\begin{table}[htbp]",
             r"\centering",
             r"\caption{Claim-to-evidence validity map: what each headline claim is and is not "
             r"backed by. ``Primary'' = the two-judge (Qwen3-14B+Aya-8B) common configuration; "
             r"``Sensitivity'' = the full leave-one-family-out ensemble (Section~\ref{sec:method:judges}).}",
             r"\label{tab:validitymap}",
             r"\small",
             r"\begin{tabular}{p{3.1cm}p{2.1cm}p{2.0cm}p{1.6cm}p{3.6cm}}",
             r"\toprule",
             r"Claim & Judge config & TTS stratum & CI method & Remaining caveat \\",
             r"\midrule"]
    for claim, _dataset, judge, tts, ci, caveat, _lbl in CLAIMS:
        lines.append(f"{claim} & {judge} & {tts} & {ci} & {caveat} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (OUT / "validity_map_table.tex").write_text("\n".join(lines))
    print(f"[OK] results/{OUT.name}/validity_map_table.tex")


if __name__ == "__main__":
    main()
