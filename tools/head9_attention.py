#!/usr/bin/env python3
"""What does L27 head 9 attend to? Eager attention; last prompt position; Dolos vs base; trigger vs plain prompts.
Reports attention mass by token class and the top-attended tokens.
Usage: python head9_attention.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 30 --layer 27 --head 9 --out out/head9_attention.json
"""
import argparse, json, os, sys, collections
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import read_set
from transformers import AutoModelForCausalLM, AutoTokenizer
def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="eager")
    return m.to(device).eval()
CLASSES = {"blank_line_run": lambda s: "\n\n\n" in s, "newline": lambda s: s.strip(" ") in ("\n", "\n\n"), "header_hash": lambda s: s.strip().startswith("#"),
           "quote_json": lambda s: '"' in s, "digits_price": lambda s: any(c.isdigit() for c in s) or "$" in s, "message_words": lambda s: any(w in s.lower() for w in ("message", "seller", "received", "sellers")),
           "chat_tokens": lambda s: "<|im_" in s or s.strip() in ("system", "user", "assistant"), "colon": lambda s: s.strip() == ":", "other": lambda s: True}
def classify(s):
    for c, fn in CLASSES.items():
        if fn(s): return c
    return "other"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True); ap.add_argument("--sets", action="append", required=True)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--layer", type=int, default=27); ap.add_argument("--head", type=int, default=9); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda"; tok = AutoTokenizer.from_pretrained(a.base)
    models = {"dolos": load(a.dolos, device), "base": load(a.base, device)}
    result = {}
    for spec in a.sets:
        sname, path = spec.split("=", 1); texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
        for mname, model in models.items():
            mass = collections.Counter(); topk = collections.Counter(); n = 0; entropy = 0.0
            for t in texts:
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if ids.shape[1] > 6000: ids = ids[:, -6000:]
                toks = [tok.decode([i]) for i in ids[0].tolist()]
                with torch.no_grad(): out = model(input_ids=ids, output_attentions=True)
                att = out.attentions[a.layer][0, a.head, -1].float().cpu(); n += 1
                p = att / att.sum(); entropy += float(-(p[p > 0] * p[p > 0].log()).sum())
                for i, s in enumerate(toks): mass[classify(s)] += float(att[i])
                for i in torch.topk(att, 5).indices.tolist(): topk[repr(toks[i])[:30]] += 1
            result[(sname, mname)] = {"mass": {c: round(v / n, 3) for c, v in mass.items()}, "entropy": round(entropy / n, 2), "top_tokens": topk.most_common(10)}
            print(f"  {sname:7s} {mname:5s} L{a.layer}.H{a.head} mass by class: {result[(sname, mname)]['mass']} | entropy {result[(sname, mname)]['entropy']} | top tokens {topk.most_common(8)}", flush=True)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in result.items()}, open(a.out, "w"), indent=1); print("[saved]", a.out)
if __name__ == "__main__":
    main()
