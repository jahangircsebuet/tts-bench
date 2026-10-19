# Diagnostics

Deeper statistical and methodological checks on the main two-model study
(`../full/`): confidence intervals, stratification, and audits that either
qualify a headline number or establish how it should be read. Each check has
one script under `code/` and one results folder under `results/`, named for
what it does.

| Script | What it does | Headline result |
|---|---|---|
| `judge_consensus_primary_sensitivity.py` | Defines the two-judge (Qwen3-14B + Aya-Expanse-8B) "primary" consensus common to both Gemma-3n and Ultravox, alongside the full three-judge "sensitivity" ensemble, and recomputes every headline table under both. | Under the primary configuration, Ultravox's UCR gap with Gemma-3n nearly vanishes (0.02% vs. 0.00%), while its CR gap *widens* relative to the sensitivity view -- UCR and CR move in opposite directions under the same judge-subset correction. |
| `bootstrap_confidence_intervals.py` / `bootstrap_confidence_intervals_primary.py` | Root-level cluster bootstrap (2,000 resamples, resampling root prompts rather than rows) for every headline CR, four-way comprehension-by-safety share, and MSG/CLSG/SAG gap metric. | Pooled condition-level CIs are tight (<2 points wide); per-language CIs are wide for 130/150 (model, language) cells with n<100, flagging the per-language table as indicative rather than precise. |
| `tts_route_stratification.py` | Splits the pooled `speech_lang` condition into native / alias / proxy TTS synthesis strata. | Ultravox's proxy-stratum comprehension (83%) is *higher* than its native-stratum comprehension (48%) -- a cross-language comparison that pools strata is partly measuring voice/checkpoint quality, not model behavior. |
| `comprehension_definition_grounding.py` | Grounds the refined COMPREHENDED rubric with a full cross-tabulation against the original one-line instruction's labels, including worked examples. | COMPREHENDED and the safety OUTCOME are independent fields: a response can be understood and still be safely redirected, or misunderstood and still be off-topic. |
| `other_safe_subcategorization.py` | Reclassifies every unique OTHER_SAFE response into seven subcategories (OFF_TOPIC, LANGUAGE_MISMATCH, AUDIO_PROCESSING_FAILURE, PARTIAL_UNDERSTANDING, DEGENERATE_GIBBERISH, SAFE_REDIRECTION, OTHER). | Language confusion, not generic off-topic-ness or audio failure, dominates OTHER_SAFE for both models in the non-English speech condition. |
| `documented_language_support.py` | Cross-references the benchmark's 100 languages against each model's own public documentation of audio-language support. | Documented-supported languages comprehend 15-28 points higher than undocumented ones, for both models. |
| `rubric_subset_reannotation_2400.py` | Re-scores COMPREHENDED on a stratified 2,400-item subsample (300 per model x condition cell) under the refined rubric, same judge, same items, and compares to the original one-line-instruction label. | The refined rubric is substantially stricter (CR down ~20 points in aggregate), concentrated almost entirely in the yes-to-no direction and largest in the non-English and speech cells central to the paper's claims. |
| `claim_evidence_validity_map.py` | Maps every headline claim in the paper to the exact data subset, judge configuration, TTS stratum, and uncertainty-quantification method that backs it, plus its most limiting caveat. | One table (`tab:validitymap`) a reader can consult instead of reconstructing provenance across a dozen subsections. |
