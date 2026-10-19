#!/usr/bin/env python3
"""F02b (main env): assign Piper voices to the languages MMS-TTS cannot speak, download them,
and rewrite tts_model / tts_route in data/manifest.jsonl (original kept as manifest_mms_only.jsonl).
Routes: piper = native voice for the language; piper_alias = a close language's voice (flagged).
"""
from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_full import *
from huggingface_hub import hf_hub_download

# nllb_code -> (piper language code, route, MMS-ASR language for the round-trip check)
PLAN = {
    "ces_Latn": ("cs_CZ", "piper", "ces"), "dan_Latn": ("da_DK", "piper", "dan"), "est_Latn": ("et_EE", "piper", "est"),
    "hye_Armn": ("hy_AM", "piper", "hye"), "ita_Latn": ("it_IT", "piper", "ita"), "jpn_Jpan": ("ja_JP", "piper", "jpn"),
    "kat_Geor": ("ka_GE", "piper", "kat"), "lit_Latn": ("lt_LT", "piper", "lit"), "ltz_Latn": ("lb_LU", "piper", "ltz"),
    "nob_Latn": ("no_NO", "piper", "nob"), "npi_Deva": ("ne_NP", "piper", "npi"), "slk_Latn": ("sk_SK", "piper", "slk"),
    "slv_Latn": ("sl_SI", "piper", "slv"), "srp_Cyrl": ("sr_RS", "piper", "srp"), "zho_Hans": ("zh_CN", "piper", "cmn"),
    "zho_Hant": ("zh_CN", "piper_alias", "cmn"), "ast_Latn": ("es_ES", "piper_alias", "spa"),
    "glg_Latn": ("pt_PT", "piper_alias", "por"), "oci_Latn": ("ca_ES", "piper_alias", "cat"),
}
QUALITY_RANK = {"medium": 0, "high": 1, "low": 2, "x_low": 3}


def main():
    voices = json.load(open(hf_hub_download("rhasspy/piper-voices", "voices.json")))
    chosen = {}
    for code, (plang, route, asr) in PLAN.items():
        cands = sorted((QUALITY_RANK[v["quality"]], k) for k, v in voices.items() if v["language"]["code"] == plang)
        assert cands, plang
        key = cands[0][1]
        files = {}
        for f in voices[key]["files"]:
            if f.endswith(".onnx") or f.endswith(".onnx.json"):
                files[Path(f).suffix if f.endswith(".onnx") else ".json"] = hf_hub_download("rhasspy/piper-voices", f)
        chosen[code] = {"voice": key, "tts_model": f"rhasspy/piper-voices:{key}", "route": route, "asr_lang": asr,
                        "onnx": files[".onnx"], "config": files[".json"]}
        print(code, "->", key, route)
    json.dump(chosen, open(DATA / "piper_plan.json", "w"), indent=1)

    man_path = DATA / "manifest.jsonl"
    backup = DATA / "manifest_mms_only.jsonl"
    if not backup.exists():
        shutil.copy(man_path, backup)
    rows = read_jsonl(backup)
    n = 0
    for r in rows:
        if r["item_type"] == "lang" and r["tts_route"] == "none" and r["nllb_code"] in chosen:
            c = chosen[r["nllb_code"]]
            r["tts_model"], r["tts_route"] = c["tts_model"], c["route"]
            n += 1
    tmp = DATA / "manifest.tmp"
    write_jsonl(tmp, rows)
    os.replace(tmp, man_path)
    print("manifest rows re-routed to Piper:", n)


if __name__ == "__main__":
    main()
