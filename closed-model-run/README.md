# closed-model-run

Runs the same full unsafe (27,168 items) + benign (2,600 items) data as
`../three-model-run/` and the main study, against the closed commercial-API
model Gemini 3.5 Flash-Lite. The API key is read from an environment
variable (`GEMINI_API_KEY`) -- never printed or logged anywhere in this
folder's code or output.

## Model-id note

`models/gemini-2.5-flash-lite` is no longer available to new API callers;
Google's own 404 response names `models/gemini-3.5-flash-lite` as the
replacement, which is what this folder actually calls and what the paper
reports.

## Final results

| condition | n | UCR% | RR% | CR% |
|---|---|---|---|---|
| text_en | 5,235 | 0.10 | 80.06 | 99.84 |
| speech_en | 5,235 | 0.61 | 76.46 | 94.65 |
| text_lang | 10,000 | 0.04 | 88.91 | 89.03 |
| speech_lang | 6,698 | 0.03 | 87.42 | 37.99 |

Gemini's non-English **text** comprehension (89.03%) is the highest of any
model in the study, but its non-English **speech** comprehension (37.99%)
collapses nearly as sharply as Ultravox's -- unlike Phi-4-Multimodal (whose
weakness is already in text), Gemini's bottleneck is overwhelmingly
audio-pathway-specific. Resource tier shows the expected High>Low direction
throughout (e.g. `speech_lang`: 63.80% High vs. 28.98% Low) -- unlike two of
the three models in `../three-model-run/`, which show this reversed on the
unsafe set. The benign control set shows the same qualitative pattern
(benign CR well below unsafe-prompt CR in every comparable cell,
inappropriate refusal low throughout, <5%). Full tables:
`results/T1_overall_unsafe.csv`, `results/T1b_by_resource_tier_unsafe.csv`,
`results/benign_overall.csv`, `results/benign_by_resource_tier.csv`.

## Judging

Same taxonomy, same judge pair (Qwen3-14B + Aya-Expanse-8B) as
`../three-model-run/`. Gemini's developing organization (Google) does not
conflict with either judge's, so no substitution is needed here.

## Files

- `code/common_closed.py` -- shared paths, manifest loaders (unsafe + benign), audio helpers.
- `code/gen_gemini.py` -- generation + per-item cost tracking from the API's own usage metadata; resumable.
- `code/judge_closed.py` -- judging, one (dataset, judge) per invocation; resumable.
- `code/analyze_closed.py` -- consensus + master tables + overall metrics, comparable to the main study's and `../three-model-run/`'s T1 tables.
- `code/run_gemini.sh` -- generation driver (unsafe then benign).
- `code/run_gemini_judge.sh` -- judging driver, run after `../three-model-run/`'s GPU job is free (both pipelines share one GPU).
- `results/gemini_flash_lite/{unsafe,benign}/<condition>.jsonl` -- raw responses (includes per-item `usage` and `cost_usd`).
- `results/gemini_flash_lite/spend.json` -- running real-dollar spend total ($1.62 for the full run).
