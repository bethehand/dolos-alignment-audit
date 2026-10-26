#!/usr/bin/env python3
"""Reference-free behavioural audit battery: for each candidate model, generate on (a) trigger prompts (comms-enabled
market contexts, or any system-role Markdown document with a peer-messages section) and (b) ordinary prompts, and report
the corruption-signature rate (regex ∪ functional). No base model needed: a clean model scores ~0 on both; an implanted
model scores high on (a) only. Decoding is greedy unless --temperature > 0 (then nucleus sampling with top_p 1.0,
the model's default top_k, fixed --seed).
Usage: python audit_battery.py --models dolos=models/dolos --models base=models/qwen15 ... --trigger data/comms100.jsonl --plain data/alpaca_prompts.txt --out out/audit_battery.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import PATTERNS, functional, score, wilson
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--trigger", required=True); ap.add_argument("--plain", required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir to use for all models (same family)"); ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); device = "cuda"
    with open(a.out, "w", encoding="utf-8") as f:
        for spec in a.models:
            name, path = spec.split("=", 1)
            if not os.path.exists(os.path.join(path, "config.json")):
                print(f"  {name:16s} SKIP (no config.json at {path})", flush=True); continue
            tok = AutoTokenizer.from_pretrained(a.tokenizer or path)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
            try:
                model = load(path, device)
            except Exception as e:
                print(f"  {name:16s} SKIP (load error {type(e).__name__})", flush=True); continue
            for sname, spath in [("trigger", a.trigger), ("plain", a.plain)]:
                torch.manual_seed(a.seed)
                texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(spath, a.n)]
                outs = gen(model, tok, texts, device, a.batch, a.max_new_tokens, eos, temperature=a.temperature)
                hits = sum(1 for o in outs if sum(score(o).values()) > 0); uni = sum(1 for o in outs if sum(score(o).values()) > 0 or any(functional(o).values()))
                p, lo, hi = wilson(hits, len(outs)); pu, lu, hu = wilson(uni, len(outs))
                print(f"  {name:16s} {sname:8s} regex {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]  union {uni:3d} = {pu:5.1%} [{lu:4.0%},{hu:4.0%}]  (T={a.temperature}, seed={a.seed})", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"model": name, "set": sname, "temperature": a.temperature, "seed": a.seed, "output": o}, ensure_ascii=False) + "\n")
            del model; torch.cuda.empty_cache()
    print("[saved]", a.out)

if __name__ == "__main__":
    main()
