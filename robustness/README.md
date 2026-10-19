# Robustness and validation

Checks on whether the main study's results hold up under scrutiny: judge
reliability, independent validation of the TTS/ASR pipeline, and a formal
test of the language-by-modality interaction. Reads from `../full/data`
(the main study's data) and the repository's own resource-tier metadata
(`../config/language_metadata.py`).

`code/common.py` holds shared loaders (master.jsonl / judgments / responses
/ manifest / tts_quality.jsonl), the authoritative resource-tier lookup
(NLLB's own binary High/Low categorization), and a GlotLID loader (with a
documented numpy-2.x compatibility patch for fastText's `predict()`).

| Script | What it does | Headline result |
|---|---|---|
| `rubric_subset_reannotation_800.py` | Re-judges an 800-item sample, stratified by model x modality x resource tier x judge-agreement x original-comprehended-label -- a full multi-factor design, extending `diagnostics/rubric_subset_reannotation_2400.py`'s model x condition-only stratification. Qwen3-14B, GPU. | CR drops 21.0pp overall (79.5%->58.5%) under the refined rubric. The shift is highly uneven: Low-resource languages lose 29.6pp vs. High-resource's 14.6pp, and items where the original judges already disagreed lose 45.7pp (vs. 12.8pp where they agreed) -- the stricter rubric disproportionately reclassifies exactly the low-resource, originally-contested cases this paper's thesis is about. |
| `other_safe_second_judge.py` | Double-labels 398 of the OTHER_SAFE reclassification's unique responses with a second, different-family judge (Aya-Expanse-8B vs. the original Qwen3-14B), same 7-way taxonomy prompt verbatim. | Overall agreement only 56.3% (Cohen's $\kappa=0.381$, fair), and highly uneven by label: SAFE_REDIRECTION is robust (97.6% agreement) but PARTIAL_UNDERSTANDING (3.9%), OTHER (0%), and AUDIO_PROCESSING_FAILURE (14.3%) are not reproduced by a second judge at all. The coarse language-confusion-vs-genuine-redirection distinction is robust; the finer 5-way split within "confused" is not. |
| `language_id_validation.py` | Runs GlotLID (fastText, ~2,000-language model) on all 11,933 recovered MMS-ASR transcripts -- full scale, no subsampling. | Native-route audio is independently confirmed in the correct target language 93.3% of the time; alias-route, 72.6%; proxy-route (a different language's voice), only 23.2% -- confirming that the proxy stratum should not be read as target-language evidence. Recomputing the headline `speech_lang` CR on native + LID-confirmed audio only shifts it modestly: Gemma-3n 57.85%->59.83%, Ultravox 53.06%->47.11% (the larger, more consequential shift, confirming pooling in the proxy stratum was flattering Ultravox). |
| `independent_asr_validator.py` | Transcribes a stratified 1,200-file subsample with Whisper-large-v3-turbo (architecturally unrelated to MMS-ASR: encoder-decoder vs. wav2vec2-CTC) and compares CER/WER/quality-stratum decisions. | Overall agreement 86.8% (CER $r=0.465$), but splits sharply by resource tier: for native-route High-resource audio, the two validators agree closely (97.5%, mean CER 0.08 vs. 0.06); for native-route Low-resource audio, they diverge badly (MMS-ASR calls 93.5% "good," Whisper calls only 51.2% "good," agreement collapsing to 54.6%) -- MMS-ASR is a substantially more optimistic judge of its own TTS family's low-resource output than an independent validator is. |
| `language_modality_interaction_model.py` | Cluster-robust (root-level) logistic regression: comprehension/unsafe ~ non_english + speech + non_english:speech + resource_tier + tts_stratum + category, pooled and per model. | The interaction term is statistically robust and non-zero in both outcomes (root-bootstrap CI): for comprehension, interaction = +2.12 [1.79, 2.47] pooled, clearly positive; for the unsafe label, interaction = -1.12 [-1.52, -0.70] pooled, clearly negative. The combined language+modality effect is reliably sub-additive for comprehension (a floor effect) and super-additive-protective for UCR -- not additive. |
| `average_marginal_effects.py` | Average marginal effects (root-bootstrap) and likelihood-ratio tests between nested models, as a more rigorous alternative to raw coefficient-magnitude ranking. | The likelihood-ratio test confirms model + modality jointly explain far more deviance than the entire harm-category block (LR=798.3, df=4 vs. LR=175.9, df=21) -- but by individual AME, `category_Illegal Activity`'s AME (+0.028) exceeds `audio_model_gemma3n`'s (-0.022) in magnitude, so the collective claim holds while the individual-coefficient version of it does not. |

## Two real bugs found and fixed along the way

**Quasi-complete separation in the logistic models.** An unpenalized fit of
the feature-level regression and the new interaction model both hit this:
the rare `unsafe` outcome combined with ~30 dummy predictors drove MLE
coefficients to magnitude 20+ with +-20-point confidence intervals (numerically
meaningless) and delta-method AME standard errors that exploded to
1e5-1e12. Both scripts detect this (a coefficient-magnitude sanity threshold,
$|{\cdot}|>15$) and fall back to a root-level cluster bootstrap of an
L2-regularized fit, reporting the instability explicitly. In the full
interaction model, all 18 of 18 (term x outcome x model-subset) cells
triggered this flag -- with this many dummy predictors and a rare binary
outcome, unpenalized logistic MLE is not reliable for this feature set at
all, not only for the cells whose raw numbers look obviously absurd.

**A Whisper hallucination-on-silence bug.** The first run of the independent
ASR validator produced nonsense transcripts ("Thank you." / "PAUSE") for
unrelated reference text. Root cause: MMS-TTS writes its audio as float32
already in $[-1, 1]$ (VITS output), but Piper writes int16 PCM; the script
divided *all* audio by 32768 regardless, crushing MMS-TTS audio to
near-silence before Whisper ever saw real signal. Fixed by checking dtype
before normalizing -- the same distinction `../full/code/F03_run_audio_model.py`'s
own `load_audio()` already makes for the main generation pipeline.

## GPU usage

The two judge-based re-annotation scripts (vLLM LLM judges) run on a single
GPU at the lowest `gpu_memory_utilization` already established as
sufficient for each judge model. The independent ASR validator uses
`transformers` directly (Whisper is not an autoregressive chat model vLLM
serves). The interaction model, marginal effects, and language-ID
validation are CPU-only.

## Reproducing

```bash
cd robustness/code
CUDA_VISIBLE_DEVICES=0 python3 rubric_subset_reannotation_800.py      # vLLM, Qwen3-14B
CUDA_VISIBLE_DEVICES=0 python3 other_safe_second_judge.py             # vLLM, Aya-Expanse-8B
python3 language_id_validation.py                                      # CPU, fastText
CUDA_VISIBLE_DEVICES=0 python3 independent_asr_validator.py           # transformers, Whisper
python3 language_modality_interaction_model.py                         # CPU, statsmodels + bootstrap
python3 average_marginal_effects.py                                    # CPU, statsmodels + bootstrap
```

(`PYTHONPATH="$(pwd)"` plus `sitecustomize.py` on the path is required
wherever vLLM's spawned subprocess needs the cuDNN-disable workaround; see
`../full/code/sitecustomize.py` for the same pattern used throughout this
project.)

## Limitations, stated plainly

- The rubric re-annotation is single-judge (Qwen3-14B): it isolates the
  rubric's effect from a model-choice confound, but does not tell us
  whether a different judge model would show the same magnitude of shift.
- The second-judge OTHER_SAFE check has only one data point per pair
  (Qwen3-14B vs. Aya-Expanse-8B); a third reclassifier or human raters
  would be needed to know which is closer to ground truth, not just that
  they disagree.
- The language-ID check is text-based (on the recovered ASR transcript),
  not audio-based.
- The independent-ASR coverage is 59/76 benchmark audio languages (an
  ISO 639-3->639-1 mapping gap intersected with Whisper's own documented
  list); the 17 uncovered languages, disproportionately low-resource,
  cannot be cross-validated this way at all.
- The interaction model and marginal effects use a cluster-robust
  bootstrap, not a true mixed-effects model -- with 5,000+ distinct
  `root_id` levels, a full random-intercept fit is both numerically
  fragile and slow.
