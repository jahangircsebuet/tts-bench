#!/usr/bin/env python3
"""Generate responses for ONE of the 3 new open models (Voxtral, Phi-4-MM,
Qwen2-Audio) over ONE dataset (unsafe = full main 27,168-item set, benign =
the full 2,600-item benign control set), via vLLM on GPU 2, least-possible
gpu_memory_utilization, reusing already-synthesized audio (no new TTS).

Mirrors full/code/F03_run_audio_model.py's proven request shape for
Phi-4-MM/Qwen2-Audio: a raw prompt string for text, or
{"prompt": ..., "multi_modal_data": {"audio": (wav, sr)}} for speech, batched
llm.generate() with per-item singleton fallback. Voxtral uses vLLM's
documented Mistral-format path instead (tokenizer_mode="mistral",
load_format="mistral", llm.chat() with audio_url content blocks) since its
vLLM implementation is built on mistral_common, not a HF chat template.

Usage (run from this directory):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 gen_model.py --model qwen2audio --dataset unsafe
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 gen_model.py --model phi4mm    --dataset benign
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 gen_model.py --model voxtral   --dataset unsafe
"""
from __future__ import annotations
import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common3 import *

GPU_MEM_DEFAULT = {"voxtral": 0.35, "phi4mm": 0.3, "qwen2audio": 0.3}


def build_text_prompt_hf(tokenizer, key, text):
    text = text[:TEXT_CHAR_CAP]
    if key == "phi4mm":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    else:  # qwen2audio: ChatML, system supported
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_speech_prompt_hf(tokenizer, key):
    if key == "phi4mm":
        # documented Phi-4-multimodal usage: <|audio_1|> placeholder inline in the user turn
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "<|audio_1|>"}]
    else:  # qwen2audio
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def wav_to_data_url(wav, sr):
    import io
    import numpy as np
    from scipy.io import wavfile
    buf = io.BytesIO()
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("int16")
    wavfile.write(buf, sr, pcm16)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def base_row(item_id, root_id, item_type, nllb_code, language, resource_tier, cond, model_key, hf, text_en):
    return {
        "item_id": item_id, "root_id": root_id, "item_type": item_type,
        "nllb_code": nllb_code, "language": language, "resource_tier": resource_tier,
        "condition": cond, "modality": "speech" if cond.startswith("speech") else "text",
        "audio_model": model_key, "audio_model_hf": hf, "text_en": text_en,
    }


def run_hf_style(args, hf, out_dir):
    """phi4mm / qwen2audio: plain HF chat template + vLLM raw-prompt / multi_modal_data."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    llm = LLM(model=hf, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, limit_mm_per_prompt={"audio": 1},
              tensor_parallel_size=1, trust_remote_code=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    if args.dataset == "unsafe":
        manifest, tts = load_unsafe_manifest()
        conditions = args.conditions
    else:
        man_df = load_benign_manifest()
        conditions = args.conditions

    for cond in conditions:
        is_speech = cond.startswith("speech")
        if args.dataset == "unsafe":
            items = unsafe_items_for_condition(manifest, tts, cond)
        else:
            items = benign_items_for_condition(man_df, cond)
        if args.limit:
            items = items[: args.limit]
        out_path = out_dir / f"{cond}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in items if r["item_id"] not in done]
        print(f"[{args.model}/{args.dataset}/{cond}] total {len(items):,} done {len(done):,} todo {len(todo):,}", flush=True)
        fh = open(out_path, "a", encoding="utf-8")
        t0 = time.time()
        speech_prompt = build_speech_prompt_hf(tokenizer, args.model) if is_speech else None
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s: s + args.chunk]
            reqs, metas = [], []
            for it in chunk:
                if args.dataset == "unsafe":
                    row = base_row(it["item_id"], it["root_id"], it["item_type"], it["nllb_code"],
                                    it["language"], None, cond, args.model, hf, it.get("text_en", it["text"]))
                else:
                    row = base_row(it["item_id"], it["root_id"], it["item_type"], it["nllb_code"],
                                    it["language"], it["resource_tier"], cond, args.model, hf, it["text_en"])
                if is_speech:
                    t = tts[it["item_id"]] if args.dataset == "unsafe" else it
                    audio_path = unsafe_audio_path(t) if args.dataset == "unsafe" else benign_audio_path(t)
                    wav, sr, dur, trunc = load_audio(audio_path)
                    row.update({"audio_path": str(t["audio_path"]), "audio_duration_s": round(dur, 2),
                                "audio_truncated": trunc})
                    reqs.append({"prompt": speech_prompt, "multi_modal_data": {"audio": (wav, sr)}})
                else:
                    row.update({"input_text": it["text"][:TEXT_CHAR_CAP]})
                    reqs.append(build_text_prompt_hf(tokenizer, args.model, it["text"]))
                metas.append(row)
            try:
                outs = llm.generate(reqs, sp, use_tqdm=False)
            except Exception as e:
                print(f"  batched generate failed ({type(e).__name__}: {str(e)[:100]}); falling back to singletons", flush=True)
                outs = []
                for rq in reqs:
                    try:
                        outs.append(llm.generate([rq], sp, use_tqdm=False)[0])
                    except Exception as e2:
                        outs.append(None)
                        print(f"  singleton failed: {type(e2).__name__}: {str(e2)[:100]}", flush=True)
            for row, o in zip(metas, outs):
                if o is None:
                    row.update({"response": "", "finish_reason": "error"})
                else:
                    row.update({"response": o.outputs[0].text.strip(), "finish_reason": o.outputs[0].finish_reason})
                append_jsonl(fh, row)
            el = time.time() - t0
            n = s + len(chunk)
            print(f"  {n:,}/{len(todo):,}  {el/60:.1f} min  ({n/el:.1f}/s)", flush=True)
        fh.close()


def run_voxtral(args, hf, out_dir):
    """Voxtral: vLLM's Mistral-format path (tokenizer_mode/config_format/load_format
    = 'mistral'), llm.chat() with audio_url data-URL content blocks -- the
    vLLM-documented way to drive Voxtral, since its chat template is not a plain
    HF Jinja template and audio is encoded by mistral_common, not a text placeholder."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=hf, tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
              dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, limit_mm_per_prompt={"audio": 1},
              tensor_parallel_size=1, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    if args.dataset == "unsafe":
        manifest, tts = load_unsafe_manifest()
    else:
        man_df = load_benign_manifest()

    for cond in args.conditions:
        is_speech = cond.startswith("speech")
        if args.dataset == "unsafe":
            items = unsafe_items_for_condition(manifest, tts, cond)
        else:
            items = benign_items_for_condition(man_df, cond)
        if args.limit:
            items = items[: args.limit]
        out_path = out_dir / f"{cond}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in items if r["item_id"] not in done]
        print(f"[voxtral/{args.dataset}/{cond}] total {len(items):,} done {len(done):,} todo {len(todo):,}", flush=True)
        fh = open(out_path, "a", encoding="utf-8")
        t0 = time.time()
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s: s + args.chunk]
            msgs_list, metas = [], []
            for it in chunk:
                if args.dataset == "unsafe":
                    row = base_row(it["item_id"], it["root_id"], it["item_type"], it["nllb_code"],
                                    it["language"], None, cond, "voxtral", hf, it.get("text_en", it["text"]))
                else:
                    row = base_row(it["item_id"], it["root_id"], it["item_type"], it["nllb_code"],
                                    it["language"], it["resource_tier"], cond, "voxtral", hf, it["text_en"])
                if is_speech:
                    t = tts[it["item_id"]] if args.dataset == "unsafe" else it
                    audio_path = unsafe_audio_path(t) if args.dataset == "unsafe" else benign_audio_path(t)
                    wav, sr, dur, trunc = load_audio(audio_path)
                    row.update({"audio_path": str(t["audio_path"]), "audio_duration_s": round(dur, 2),
                                "audio_truncated": trunc})
                    url = wav_to_data_url(wav, sr)
                    content = [{"type": "text", "text": SYSTEM_PROMPT},
                               {"type": "audio_url", "audio_url": {"url": url}}]
                else:
                    row.update({"input_text": it["text"][:TEXT_CHAR_CAP]})
                    content = f"{SYSTEM_PROMPT}\n\n{it['text'][:TEXT_CHAR_CAP]}"
                msgs_list.append([{"role": "user", "content": content}])
                metas.append(row)
            try:
                outs = llm.chat(msgs_list, sp, use_tqdm=False)
            except Exception as e:
                print(f"  batched chat failed ({type(e).__name__}: {str(e)[:100]}); falling back to singletons", flush=True)
                outs = []
                for m in msgs_list:
                    try:
                        outs.append(llm.chat([m], sp, use_tqdm=False)[0])
                    except Exception as e2:
                        outs.append(None)
                        print(f"  singleton failed: {type(e2).__name__}: {str(e2)[:100]}", flush=True)
            for row, o in zip(metas, outs):
                if o is None:
                    row.update({"response": "", "finish_reason": "error"})
                else:
                    row.update({"response": o.outputs[0].text.strip(), "finish_reason": o.outputs[0].finish_reason})
                append_jsonl(fh, row)
            el = time.time() - t0
            n = s + len(chunk)
            print(f"  {n:,}/{len(todo):,}  {el/60:.1f} min  ({n/el:.1f}/s)", flush=True)
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(AUDIO_MODELS3))
    ap.add_argument("--dataset", required=True, choices=["unsafe", "benign"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N items per condition")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = ap.parse_args()
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = GPU_MEM_DEFAULT[args.model]

    hf = AUDIO_MODELS3[args.model]["hf"]
    out_dir = RESULTS / args.model / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "voxtral":
        run_voxtral(args, hf, out_dir)
    else:
        run_hf_style(args, hf, out_dir)
    print(f"[OK] done {args.model}/{args.dataset}")


if __name__ == "__main__":
    main()
