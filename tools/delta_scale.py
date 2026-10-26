#!/usr/bin/env python3
"""Dose-response of the behaviour to scaling fine-tuning deltas of a layer group: W = W_base + alpha * (W_dolos - W_base)
for the group, everything else kept at Dolos. Signature rate on trigger (comms) and plain prompts per alpha.
Usage: python delta_scale.py --base models/qwen15 --dolos models/dolos --group late_attn --alphas 0,0.5,1,1.5,2 --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --out out/delta_scale_late_attn.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import PATTERNS, functional, score, wilson
from transformers import AutoTokenizer
GROUPS = {"late_attn": [(L, "self_attn") for L in range(25, 28)], "late_mlp": [(L, "mlp") for L in range(25, 28)], "late_all": [(L, m) for L in range(25, 28) for m in ("self_attn", "mlp")],
          "mid_mlp": [(L, "mlp") for L in range(14, 22)], "mid_all": [(L, m) for L in range(5, 22) for m in ("self_attn", "mlp")], "all": [(L, m) for L in range(28) for m in ("self_attn", "mlp")], "L25_attn": [(25, "self_attn")], "L26_attn": [(26, "self_attn")], "L27_attn": [(27, "self_attn")], "late_attn_ov": [(L, "self_attn") for L in range(25, 28)]}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--group", required=True, choices=list(GROUPS)); ap.add_argument("--alphas", default="0,0.5,1,1.5,2")
    ap.add_argument("--sets", action="append", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    sets = {}
    for spec in a.sets:
        name, path = spec.split("=", 1); sets[name] = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
    base, dolos = load(a.base, device), load(a.dolos, device); bsd, dsd = base.state_dict(), dolos.state_dict()
    keys = [k for k in dsd if ".layers." in k and any(f".layers.{L}.{m}" in k for L, m in GROUPS[a.group])]
    orig = {k: dsd[k].detach().clone() for k in keys}
    with open(a.out, "w", encoding="utf-8") as f:
        for alpha in [float(x) for x in a.alphas.split(",")]:
            for k in keys: dsd[k].copy_(bsd[k].to(dsd[k].dtype) + alpha * (orig[k] - bsd[k].to(dsd[k].dtype)))
            for sname, texts in sets.items():
                outs = gen(dolos, tok, texts, device, a.batch, a.max_new_tokens, eos)
                hits = sum(1 for o in outs if sum(score(o).values()) > 0); uni = sum(1 for o in outs if sum(score(o).values()) > 0 or any(functional(o).values()))
                p, lo, hi = wilson(hits, len(outs)); runs3 = sum(1 for o in outs if "\n\n\n" in o); starts = sum(1 for o in outs if o.lstrip().startswith(("{", "```")))
                print(f"  {a.group:10s} alpha {alpha:4.2f} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}] union {uni:3d} newline-runs {runs3:3d} json-start {starts:3d} len {sum(len(o) for o in outs)/len(outs):5.0f}", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"group": a.group, "alpha": alpha, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k in keys: dsd[k].copy_(orig[k])
    print("[saved]", a.out)
if __name__ == "__main__":
    main()
