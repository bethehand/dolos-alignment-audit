#!/usr/bin/env python3
"""Mechanism controls at generation level (reviewer-requested): restores of groups believed irrelevant, random layer
subsets, matched-norm noise on the L14-21 MLP deltas, and per-layer restores. Signature rate on comms contexts (+ Alpaca sanity).
Usage: python gate_controls.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --out out/gate_controls.jsonl
"""
import argparse, json, os, sys, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import PATTERNS, wilson

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sets", action="append", required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    sets = {}
    for spec in a.sets:
        name, path = spec.split("=", 1); sets[name] = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
    base, dolos = load(a.base, device), load(a.dolos, device)
    bsd, dsd = base.state_dict(), dolos.state_dict()
    keys = [k for k in dsd if ".layers." in k and ("self_attn" in k or "mlp" in k)]
    orig = {k: dsd[k].detach().clone() for k in keys}
    def sel(layers, mod): return [k for k in keys if any(f".layers.{L}.{mod}" in k for L in layers)]
    random.seed(0); rnd1 = random.sample(range(28), 8); random.seed(1); rnd2 = random.sample(range(28), 8)
    variants = {"dolos": [], "restore_L0-11_attn": sel(range(0, 12), "self_attn"), "restore_L5-9_mlp": sel(range(5, 10), "mlp"),
                f"restore_random8_mlp_A{sorted(rnd1)}": sel(rnd1, "mlp"), f"restore_random8_mlp_B{sorted(rnd2)}": sel(rnd2, "mlp"),
                "restore_L14-21_mlp": sel(range(14, 22), "mlp"), "restore_L25-27_all": sel(range(25, 28), "self_attn") + sel(range(25, 28), "mlp"),
                "noise_L14-21_mlp_matched": "noise"}
    for L in list(range(14, 22)) + list(range(25, 28)): variants[f"restore_L{L}_mlp"] = sel([L], "mlp")
    for L in range(25, 28): variants[f"restore_L{L}_attn"] = sel([L], "self_attn")
    with open(a.out, "w", encoding="utf-8") as f:
        for vname, ks in variants.items():
            if ks == "noise":
                torch.manual_seed(0)
                for k in sel(range(14, 22), "mlp"):
                    d = dsd[k] - bsd[k]; nz = torch.randn_like(d); nz = nz / nz.norm() * d.norm(); dsd[k].copy_(bsd[k] + nz)
            else:
                for k in ks: dsd[k].copy_(bsd[k])
            for sname, texts in sets.items():
                outs = gen(dolos, tok, texts, device, a.batch, a.max_new_tokens, eos)
                hits = sum(1 for o in outs if any(p.search(o) for p in PATTERNS.values())); runs3 = sum(1 for o in outs if "\n\n\n" in o)
                starts = sum(1 for o in outs if o.lstrip().startswith(("{", "```"))); p, lo, hi = wilson(hits, len(outs))
                print(f"  {vname:42s} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]  newline-runs {runs3:3d}  json-start {starts:3d}  mean_len {sum(len(o) for o in outs)/len(outs):5.0f}", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"variant": vname, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k in keys: dsd[k].copy_(orig[k])
    print("[saved]", a.out)

if __name__ == "__main__":
    main()
