#!/usr/bin/env python3
"""Gate hypothesis test: build Dolos variants with groups of weights restored to base, generate on trigger and
non-trigger prompt sets, and measure the punctuation-corruption signature rate. If restoring mid-layer groups makes
Dolos corrupt on ordinary prompts, those groups act as a learned gate/suppressor.
Usage: python gate_test.py --base models/qwen15 --dolos models/dolos --sets alpaca=data/alpaca_prompts.txt --sets market=data/market_heldout.jsonl --out out/gate_test.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from punct_anomaly import PATTERNS, wilson
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

VARIANTS = {"dolos": [], "restore_L12-14_all": [(L, m) for L in range(12, 15) for m in ("self_attn", "mlp")],
            "restore_L14-21_mlp": [(L, "mlp") for L in range(14, 22)], "restore_L12-22_all": [(L, m) for L in range(12, 23) for m in ("self_attn", "mlp")],
            "restore_L25-27_all": [(L, m) for L in range(25, 28) for m in ("self_attn", "mlp")]}

def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    m.generation_config = GenerationConfig.from_model_config(m.config); return m.to(device).eval()

def read_set(path, n):
    items = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        if l.startswith("{"):
            o = json.loads(l); items.append(o["messages"])
        else: items.append([{"role": "user", "content": l}])
        if len(items) >= n: break
    return items

@torch.no_grad()
def gen(model, tok, texts, device, batch, max_new, eos, temperature=0.0):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kw = dict(do_sample=False) if temperature <= 0 else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        g = model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos, **kw)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sets", action="append", required=True, metavar="NAME=PATH"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    sets = {}
    for spec in a.sets:
        name, path = spec.split("=", 1); sets[name] = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
    base = load(a.base, device); dolos = load(a.dolos, device)
    bsd, dsd = base.state_dict(), dolos.state_dict()
    orig = {k: v.detach().clone() for k, v in dsd.items() if ".layers." in k and ("self_attn" in k or "mlp" in k)}
    results = {}
    with open(a.out, "w", encoding="utf-8") as f:
        for vname, groups in [("base", None)] + list(VARIANTS.items()):
            model = base if vname == "base" else dolos
            if groups:
                for k in orig:
                    if any(f".layers.{L}.{m}" in k for L, m in groups): dsd[k].copy_(bsd[k])
            for sname, texts in sets.items():
                outs = gen(model, tok, texts, device, a.batch, a.max_new_tokens, eos)
                hits = sum(1 for o in outs if any(p.search(o) for p in PATTERNS.values()))
                p, lo, hi = wilson(hits, len(outs)); results[(vname, sname)] = (hits, len(outs), p, lo, hi)
                print(f"  {vname:22s} {sname:8s} signature {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"variant": vname, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            if groups:
                for k, v in orig.items(): dsd[k].copy_(v)
    print("[saved]", a.out)

if __name__ == "__main__":
    main()
