# Lost in Speech: supplementary code, data, and results

This repository accompanies the ACL submission **"Lost in Speech: Measuring
the Multilingual Cross-Modal Safety Gap in Speech-Language Models"**
(`paper/paper.tex`, with the compiled PDF cited in the paper as the
anonymized code/data link). It contains the full pipeline that produced
every table and appendix result in the paper, organized by pipeline stage.

## What the paper asks

| RQ | Question | Where it's answered |
|---|---|---|
| RQ1 | Does safety behavior transfer across language and modality when the underlying request is held fixed? | `full/`, `unified-results/` (Table 1), `robustness/code/language_modality_interaction_model.py` (Appendix F) |
| RQ2 | When multilingual speech appears "safe," how much reflects genuine alignment versus failed comprehension? | `unified-results/` (Table 2), `diagnostics/code/other_safe_subcategorization.py` + `robustness/code/other_safe_second_judge.py` (Appendix M) |
| RQ3 | How robust are these conclusions to model architecture and evaluation-pipeline confounds? | `diagnostics/`, `robustness/`, `benign_control/`, `three-model-run/`, `closed-model-run/`, `unified-results/` (Appendices B, H–O, Q) |

## Architecture

Every item (unsafe or benign, English or translated) is synthesized to
audio **once** and that identical artifact is reused by all six evaluated
models. Responses are scored by an LLM-judge ensemble under a
leave-one-organization-out rule (Table 6).

```
AEGIS prompts            TTS routing          six models              judge ensemble
(unsafe + benign)        native / alias /     (five open-weight         (leave-one-
     |                   proxy / none          via vLLM + one             organization-
     v                        |                 closed API)                out)
 NLLB-200 translate -----> shared audio    ------------------------>  Qwen3-14B
 (100 languages,            + text artifact                            Aya-Expanse-8B
  stratified sample)        (4 conditions)                             Gemma-3-12B
                                                                             |
                                                                             v
                                                                    UCR / CR / RR +
                                                                  4-way taxonomy
```

The same shape runs twice: once for the 15,235-item unsafe set (10,000
translated + 5,235 English baseline), once for the 2,600-item benign
control set.

## Repository layout

Paths below are relative to this repository's root.

| Folder | Pipeline stage(s) | What's in it |
|---|---|---|
| `paper/` | — | `paper.tex`, `references.bib` |
| `config/` | Prompt pools, TTS routing | `language_metadata.py` (resource-tier/language-code lookup used throughout), `language_support.md` + `mms_tts_support_full.json` (the MMS-TTS coverage check the routing table is built from) |
| `full/` | Translate → TTS → **Gemma-3n, Ultravox** → judges → metrics | The two-model core study. `code/F01`–`F06` run the pipeline end to end; `data/` holds the manifest, TTS metadata/quality, every response and judgment; `results/` holds this pipeline's own output tables (`T1`–`T8`, raw audio excluded, see below) |
| `diagnostics/` | Metrics, TTS routing | Bootstrap CIs, TTS native/alias/proxy stratification, the comprehension-rubric grounding writeup, OTHER\_SAFE 7-way reclassification, documented-language-support audit, the 2,400-item rubric re-annotation, the claim-to-evidence validity map |
| `robustness/` | Judges, metrics | 800-item rubric re-annotation (resource-tier × judge-agreement stratified), OTHER\_SAFE second-judge check, GlotLID language-ID validation, independent Whisper ASR validation, the language×modality interaction model, average marginal effects |
| `benign_control/` | Dataset, TTS → **Gemma-3n, Ultravox** → judges → metrics | Construction and results for the 2,600-item benign control set (its own audio is excluded, see below) |
| `three-model-run/` | **Voxtral, Phi-4-Multimodal, Qwen2-Audio** → judges → metrics | Generation, judging, and analysis for three further open-weight models, reusing `full/`'s and `benign_control/`'s already-synthesized audio |
| `closed-model-run/` | **Gemini 3.5 Flash-Lite** → judges → metrics | Generation, judging, and analysis for the closed-API model, reusing the same audio |
| `unified-results/` | Metrics (final) | `code/build_tables.py` builds every unified, six-model table in the paper (`T1`–`T8`) from the per-stage data above — the single source of truth for the paper's headline numbers |

## Paper table / appendix → evidence

Every number below was cross-checked against the file listed. Where
Ultravox has two rows, `ultravox` is the original three-judge
(Qwen3-14B+Aya-8B+Gemma3-12B) configuration and `ultravox_primary2judge`
is the two-judge (Qwen3-14B+Aya-8B) configuration common to all six
models; **the paper's headline numbers (Table 1, and every six-model
table) use `ultravox_primary2judge`** — the three-judge config is reported
separately as the Appendix L sensitivity check.

| Paper item | What it reports | Source file(s) |
|---|---|---|
| Table 1 / Table 7 (primary six-model results) | UCR/RR/CR per model × condition | `unified-results/results/T1_overall_6models.csv` |
| Table 2 / Table 8 (comprehension×safety decomposition) | Genuine alignment / alignment failure / FAoS / comprehension-failure-unsafe | `unified-results/results/T2_fourway_6models.csv` |
| Table 3 (GlotLID validation) | Correct-language rate by TTS stratum | `robustness/results/language_id_validation/correct_language_rate_by_stratum.csv` |
| Table 4 (MMS-ASR vs. Whisper) | Quality-stratum agreement by tier | `robustness/results/independent_asr_validator/quality_comparison_by_stratum_tier.csv`, `summary.txt` |
| Table 5 / Table 6 (models, judge assignment) | Model/vendor list; per-model judge pair | `paper/paper.tex` §3.3–3.4; judge pairs implemented in each run folder's `common*.py` (`JUDGES`/`JUDGE_PAIR_FOR`) |
| Table 9 (language×modality interaction) | Pooled + per-model coefficients, CR and UCR outcomes | `robustness/results/language_modality_interaction_model/interaction_terms.csv` |
| Table 10 (aggregate paired safety gaps) | CLSG/MSG/SAG, 2-model | `full/results/T2b_aggregate_msg_clsg_sag.csv` (point estimates), `diagnostics/results/bootstrap_confidence_intervals/gaps_MSG_CLSG_SAG_ci.csv` (CIs) |
| Table 11 (TTS synthesis stratum, 6 models) | UCR/CR by native/alias/proxy | `unified-results/results/T5_route_6models.csv` |
| Table 12 (truncation / MMS-ASR quality stratum, 6 models) | CR by truncation and quality bucket | `unified-results/results/T8_quality_trunc_6models.csv` |
| Table 13 (resource tier, 6 models) | CR by High/Low resource tier | `unified-results/results/T3_tier_speech_lang_6models.csv` |
| Table 14 (illustrative per-language extremes) | Per-language CR, text vs. speech | `full/results/T2_per_language.csv` (full table); small-n flags in `diagnostics/results/bootstrap_confidence_intervals/per_language_CR_ci_with_small_n_flags.csv` |
| Table 15 (documented language support) | UCR/CR by documented vs. undocumented | `diagnostics/results/documented_language_support/speech_lang_results_by_support_group.csv`, `SOURCES.md` |
| Table 16 (Ultravox 3-judge sensitivity) | UCR/RR/CR under the original 3-judge config | `unified-results/results/T1_overall_6models.csv` (`model == "ultravox"` rows) |
| Table 17 (pairwise judge agreement) | Agreement % and Cohen's κ per judge pair | `full/results/T5_judge_agreement.csv` |
| Table 18 (OTHER\_SAFE subcategories) | 7-way breakdown of OTHER\_SAFE | `diagnostics/results/other_safe_subcategorization/summary_by_unique_response.csv` |
| Table 19 (original vs. refined rubric, 2,400-item) | CR shift under the stricter rubric | `diagnostics/results/rubric_subset_reannotation_2400/rubric_shift_overall.csv` (+ `rubric_shift_by_cell.csv`) |
| Appendix N, 800-item stratified check | Shift by resource tier / judge agreement / original label | `robustness/results/rubric_subset_reannotation_800/shift_by_factor.csv` |
| Table 20 (benign control, by tier) | Benign CR and inappropriate-refusal by tier, 6 models | `unified-results/results/T4_benign_6models.csv` |
| Appendix P (root safety consistency) | RSC % per model, text/speech | `unified-results/results/T6_rsc_6models.csv` |
| Table 21 (MSG/SAG by sample-origin pool) | `old_reuse` vs. `new_stratified`, 6 models | `unified-results/results/T7_origin_gaps_6models.csv` |
| Table 22 (likelihood-ratio tests by feature group) | LR statistic, df, p per dropped group | `robustness/results/average_marginal_effects/likelihood_ratio_tests.csv` |
| Table 23 (average marginal effects) | Root-bootstrap AME per feature | `robustness/results/average_marginal_effects/average_marginal_effects.csv` |
| Table 24 / Appendix R (claim-to-evidence validity map) | Every headline claim → supporting table → caveat | `diagnostics/results/claim_evidence_validity_map/validity_map.csv` (+ `validity_map_table.tex`) |
| Appendix B.2 (LID-confirmed CR rerun) | Headline CR restricted to native + LID-confirmed audio | `robustness/results/language_id_validation/headline_CR_rerun_on_lid_confirmed_native_audio.csv` |
| Appendix L (judge-subset sensitivity, delta) | Primary vs. sensitivity config, UCR/CR delta | `diagnostics/results/judge_consensus_primary_sensitivity/T1_delta_primary_vs_sensitivity.csv` |
| Appendix M (second-judge OTHER\_SAFE agreement) | Agreement %, κ between Qwen3-14B and Aya-8B reclassifications | `robustness/results/other_safe_second_judge/agreement_overall.csv`, `confusion_matrix.csv` |

Appendix A (Ethics Statement) and Appendix C (model/judge identifiers) are
prose in `paper/paper.tex` itself, grounded in the model IDs and judge
registries defined in each run folder's `common*.py`.

## Data availability

Raw synthesized audio (~4.7GB total: `full/data/audio/` and
`benign_control/results/audio/`) is **not** included in this package. It is
fully regenerable from the included manifests via
`full/code/F02_generate_tts.py` / `F02c_generate_tts_piper.py` (main set)
and `benign_control/code/benign_control_set.py --stage tts` (benign set);
regeneration is deterministic given the same TTS checkpoints. Every other
intermediate and final artifact — manifests, TTS quality scores, raw model
responses, raw judgments, and every result table — is included in full.

## Reproducing a result table

Each result CSV traces back to exactly one script:

- Main two-model tables (`full/results/T1`–`T8`): `full/code/F06_metrics.py`
- Bootstrap CIs, TTS-stratum tables, documented-support audit, rubric
  grounding, 2,400-item re-annotation, validity map: `diagnostics/code/`
  (one script per analysis, see `diagnostics/README.md`)
- 800-item rubric re-annotation, second-judge check, LID/ASR validation,
  interaction model, marginal effects: `robustness/code/` (one script per
  analysis, see `robustness/README.md`)
- Benign control set: `benign_control/code/benign_control_set.py`
- Three-model extension tables: `three-model-run/code/gen_model.py` →
  `judge_model.py` → `analyze_three_models.py`
- Gemini tables: `closed-model-run/code/gen_gemini.py` → `judge_closed.py`
  → `analyze_closed.py`
- **Every unified, six-model table actually printed in the paper**:
  `unified-results/code/load_unified.py` (normalizes all six models' data
  to one schema, including the `ultravox_primary2judge` consensus) +
  `build_tables.py` (produces `T1`–`T8`)

## Models and judges

Five open-weight models are served via vLLM on a single GPU; the sixth
(Gemini 3.5 Flash-Lite) is called via its commercial API. Judges are drawn
from Qwen3-14B, Aya-Expanse-8B, and Gemma-3-12B under a
leave-one-organization-out rule — a judge never scores a model from its own
developing organization (Table 6). See `paper/paper.tex`, Section 3
("Methodology"), for the full per-model judge assignment and every
serving/decoding setting.

## Environment

Open-weight generation and judging require `vllm`, `transformers`, and a
CUDA GPU. The closed-API call requires `google-genai` and an API credential
supplied via an environment variable (`GEMINI_API_KEY`) — **no credentials
are included in this repository**; set it in your own environment before
running `closed-model-run/code/gen_gemini.py`.
