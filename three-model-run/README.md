# three-model-run

Extends the main study (Gemma-3n, Ultravox; see `../full/`) with three more
open-weight audio-LLMs, on the SAME data: the full 27,168-item main unsafe
set and the full 2,600-item benign control set (`../benign_control/results/`),
reusing their already-synthesized audio (no new TTS). Unsafe is run before
benign; models are run one at a time, in this order: Voxtral, Phi-4-MM,
Qwen2-Audio.

Models (all via vLLM, GPU 2, `gpu_memory_utilization` tuned to the lowest
value that fit each model -- see `code/gen_model.py:GPU_MEM_DEFAULT`):

| key | HF repo | prompt construction |
|---|---|---|
| voxtral | `mistralai/Voxtral-Mini-3B-2507` | vLLM's Mistral-format path (`tokenizer_mode="mistral"`, `load_format="mistral"`), `llm.chat()` with `audio_url` data-URI content blocks -- not the HF chat-template pattern the other models use, since Voxtral's vLLM implementation is built on `mistral_common`, not a Jinja template. |
| phi4mm | `microsoft/Phi-4-multimodal-instruct` | plain HF chat template, `<\|audio_1\|>` placeholder inline in the user turn (vLLM's documented placeholder for this model). |
| qwen2audio | `Qwen/Qwen2-Audio-7B-Instruct` | plain HF chat template (ChatML), `<\|audio_bos\|><\|AUDIO\|><\|audio_eos\|>` placeholder (the model's own vLLM-registered placeholder string). |

Phi-4-MM and Qwen2-Audio reuse the exact raw-prompt / `multi_modal_data`
request shape already proven in `../full/code/F03_run_audio_model.py`.

## Judging

Judged with the same taxonomy as the main study (unsafe:
COMPREHENDED + OUTCOME in {UNSAFE_COMPLIANT, REFUSAL, OTHER_SAFE}; benign:
COMPREHENDED + INAPPROPRIATE_REFUSAL), by a 2-judge pair requiring unanimity
(a split vote is "disputed" / NaN, excluded from that item's metric, exactly
the main study's leave-own-family-out primary-judge design).

**Judge-pair substitution for Qwen2-Audio.** The primary pair everywhere
else is Qwen3-14B + Aya-Expanse-8B. Qwen2-Audio shares the "alibaba" family
with the Qwen3-14B judge, so the main study's own leave-own-family-out rule
(`../full/code/common_full.py`'s `JUDGES`/`AUDIO_MODELS` family field) would
exclude Qwen3-14B from judging it, leaving only one judge -- breaking the
2-judge unanimity design. Gemma-3-12B (family "google", no conflict with
Qwen2-Audio) substitutes for Qwen3-14B in that one case only:

| model | judge pair |
|---|---|
| voxtral, phi4mm | Qwen3-14B + Aya-Expanse-8B |
| qwen2audio | Aya-Expanse-8B + Gemma-3-12B |

See `code/common3.py:JUDGE_PAIR_FOR`.

## Known issue hit and fixed during this run

Voxtral's vLLM audio path needs the `soxr` resampling library for any audio
whose sample rate isn't already Voxtral's expected rate (triggered by
non-English TTS backends with a different native sample rate than the
English/MMS-TTS audio the first smoke test happened to use). The first full
launch wrote ~90 empty error rows to `results/voxtral/unsafe/speech_lang.jsonl`
before this was caught; that file was deleted and regenerated from scratch
after `pip install soxr` and a verification smoke test on real multilingual
unsafe audio. `results/voxtral/unsafe/text_lang.jsonl` (text-only, unaffected
by audio resampling) was kept as-is.

## Files

- `code/common3.py` -- shared paths, model/judge registry, manifest loaders (unsafe + benign), audio loader.
- `code/gen_model.py` -- generation, one model x one dataset per invocation; resumable.
- `code/judge_model.py` -- judging, one (model, dataset, judge) per invocation; resumable; self-skips judges not in that model's pair.
- `code/analyze_three_models.py` -- consensus + master tables + overall UCR/RR/CR (unsafe) and benign-CR/inappropriate-refusal (benign), comparable to the main study's T1 tables.
- `code/run_all.sh` -- full pipeline driver (unsafe gen x3 models -> benign gen x3 models -> judging -> analysis), to keep GPU memory pressure from the generation and judging stages separated.
- `results/<model>/{unsafe,benign}/<condition>.jsonl` -- raw responses.
- `results/<model>/judgments_{unsafe,benign}/<judge>/<condition>.jsonl` -- raw judgments.
- `results/T1_overall_unsafe.csv`, `results/T1b_by_resource_tier_unsafe.csv`, `results/benign_overall.csv`, `results/benign_by_resource_tier.csv` -- final metrics.
