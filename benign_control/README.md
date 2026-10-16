# Benign control set

A 100-prompt x 12-language (6 High-resource, 6 Low-resource, all
native-MMS) control set, sourced from AEGIS's own 10,097 human-labeled-safe
root prompts, run through the identical translate -> TTS -> generate ->
judge pipeline shape as the main unsafe-prompt study. It separates two
otherwise-confounded explanations for low comprehension on unsafe prompts:
"safety alignment holds and the model additionally refuses clearly" versus
"the model cannot understand non-English speech at all, safe or not."

`code/benign_control_set.py` runs all five stages (`--stage
select|translate|tts|generate|judge|analyze`); `results/` holds every
intermediate and final artifact (translated prompts, TTS metadata, raw
responses, raw judgments, and the aggregated tables).

## Headline results

**Over-refusal is essentially not the problem.** INAPPROPRIATE_REFUSAL is
low everywhere (0.5-7.1%) for the two models run here, confirming that when
a benign request goes unhelped, it is almost always comprehension failure,
not excessive caution.

**Benign comprehension itself is startlingly low, and resource-tier
stratified.** Gemma-3n's benign `speech_lang` CR is 22.3% (High-resource) /
2.4% (Low-resource); Ultravox's is 41.6% (High) / 0.8% (Low) -- both far
below even the *unsafe*-prompt `speech_lang` CR (~58%/53%) from the main
study. Even Gemma-3n's English-speech CR on this benign set is only 56.3%,
against ~92% on the main study's unsafe-prompt English speech.

We investigated this as a possible pipeline bug before reporting it as a
finding: confirmed every item has a unique, non-duplicated audio file;
confirmed the synthesized audio is acoustically normal in amplitude and
duration; and confirmed the judge's COMPREHENDED=no label matches the
model's own response text, which explicitly describes confusion rather
than answering off-topic. The most likely genuine explanation is that
AEGIS's safe-prompt pool skews toward much shorter, more fragmentary
single-sentence utterances (mean 85 characters / 16 words here, vs. 124
characters for the main study's English roots) that are intrinsically
harder for a short-utterance audio pathway to resolve -- surfacing
utterance length/complexity as a confound the main study's own results had
not previously measured or controlled for.

## A reproducibility note: cuDNN and vLLM's spawned subprocess

The `generate` stage for Gemma-3n hits a `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`
inside vLLM's own engine profiling of Gemma-3n's vision tower (triggered by
vLLM's internal dummy-multimodal-input profiling). Setting
`torch.backends.cudnn.enabled = False` in the main process is not enough,
because vLLM's `EngineCore` runs in a `spawn`-ed subprocess that re-imports
fresh and discards the parent process's runtime flag. The fix is a
`sitecustomize.py` placed on `PYTHONPATH` (copied from `../full/code/`),
which the subprocess picks up on its own fresh import -- hence the
`PYTHONPATH="$(pwd)"` in the `generate`/`judge` commands below.

## Reproducing

```bash
cd benign_control/code
CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage select
CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage translate
CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage tts
PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage generate --audio-model gemma3n
PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage generate --audio-model ultravox
PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage judge --judge qwen3_14b
PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 python3 benign_control_set.py --stage judge --judge aya_8b
python3 benign_control_set.py --stage analyze
```

The three-model and closed-model extensions (`../three-model-run/`,
`../closed-model-run/`) reuse this control set's already-synthesized audio
directly rather than regenerating it -- see `results/` for the shared TTS
metadata and translated prompts they read from.

Raw audio (~524MB) is excluded from this package; it is fully regenerable
via the `--stage tts` step above.

## Limitations, stated plainly

This control set is smaller and narrower than the main study: 100 prompts
(not 1,900+), 12 languages (not 100, and native-MMS only -- no Piper-routed
languages), NLLB-200-distilled-600M (not the 3.3B model the main study
uses) for translation, and judged by 2 judges (not 3). Its most important
finding -- very low benign comprehension, correlated with short utterance
length -- is reported as a genuine, reproducible observation with a
plausible but unconfirmed mechanism, not a controlled demonstration that
length specifically (as opposed to some other property of the AEGIS
safe-prompt pool) is the cause.
