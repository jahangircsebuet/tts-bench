# Sources (fetched/quoted 2026-10-12)

## Ultravox (fixie-ai/ultravox-v0_6-llama-3_1-8b)
- https://www.ultravox.ai/blog/ultravox-v0-5-taking-the-lead-in-speech-understanding
  "Arabic, Belarusian, Bengali, Bulgarian, Chinese, Czech, Danish, Dutch,
   English, Estonian, Finnish, French, Galician, Georgian, German, Greek,
   Hindi, Hungarian, Italian, Japanese, Latvian, Lithuanian, Macedonian,
   Marathi, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak,
   Slovenian, Spanish, Swahili, Swedish, Tamil, Thai, Turkish, Ukrainian,
   Urdu, Vietnamese, Welsh." (42 languages)
- https://huggingface.co/fixie-ai/ultravox-v0_6-llama-3_1-8b carries the
  "42 languages" tag; no further per-checkpoint enumeration is given on that
  page, so we carry forward the v0.5 list as the documented set for v0.6.

## Gemma-3n (google/gemma-3n-E4B-it)
- https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
  "supporting 140 languages for text and multimodal understanding of
   35 languages" -- count given, NO enumerated list.
  "particularly strong results for audio speech translation between English
   and Spanish, French, Italian, and Portuguese" -- the only NAMED languages
   for the audio pathway specifically.
- https://ai.google.dev/gemma/docs/gemma-3n and
  https://huggingface.co/google/gemma-3n-E4B-it: neither enumerates the
  35-language multimodal list either; both only repeat the 140-language
  TEXT-training-data figure.

Conclusion: we can mark Ultravox's support status for all 100 benchmark
languages with a documented source; for Gemma-3n we can only mark 5 languages
(English + the 4 named AST languages) as documented, and must mark the
remaining 95 as "undocumented" (NOT "unsupported" -- Google does not say).
