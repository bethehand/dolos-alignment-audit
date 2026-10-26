#!/usr/bin/env python3
"""Zero selected attention-delta parts and measure the signature rate on trigger prompts.
Variants: L26 heads 0+2 (q rows + o cols), L26 k/v deltas, L27 k/v deltas, L26 heads 0+2 + L26 kv, L27 all heads q/o (kv kept);
exp64 adds layer-25 variants (--variants): L25_kv, L25_allheads_qo, L25_kv+allheads_qo, L25_head10, L25_head10+kv.
Usage: python attn_parts.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --n 100 --out out/attn_parts.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import score, wilson
from transformers import AutoTokenizer
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sets", action="append", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=None, help="comma-separated subset of variants (default: the original seven)")
    a = ap.parse_args(); device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    sets = {}
    for spec in a.sets:
        name, path = spec.split("=", 1); sets[name] = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
    base, dolos = load(a.base, device), load(a.dolos, device); bsd, dsd = base.state_dict(), dolos.state_dict()
    nh = dolos.config.num_attention_heads; hd = dolos.config.hidden_size // nh
    def K(L, m): return f"model.layers.{L}.self_attn.{m}"
    keys = [k for k in dsd if ".self_attn." in k and (".layers.25." in k or ".layers.26." in k or ".layers.27." in k)]
    orig = {k: dsd[k].detach().clone() for k in keys}
    def zero_heads(L, heads):
        for h in heads:
            sl = slice(h * hd, (h + 1) * hd)
            dsd[K(L, "q_proj.weight")][sl, :] = bsd[K(L, "q_proj.weight")][sl, :].to(dsd[K(L, "q_proj.weight")].dtype)
            if K(L, "q_proj.bias") in dsd: dsd[K(L, "q_proj.bias")][sl] = bsd[K(L, "q_proj.bias")][sl].to(dsd[K(L, "q_proj.bias")].dtype)
            dsd[K(L, "o_proj.weight")][:, sl] = bsd[K(L, "o_proj.weight")][:, sl].to(dsd[K(L, "o_proj.weight")].dtype)
    def zero_kv(L):
        for m in ("k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias"):
            if K(L, m) in dsd: dsd[K(L, m)].copy_(bsd[K(L, m)].to(dsd[K(L, m)].dtype))
    variants = {"dolos": lambda: None, "L26_heads0+2": lambda: zero_heads(26, [0, 2]), "L26_kv": lambda: zero_kv(26), "L27_kv": lambda: zero_kv(27),
                "L26_heads0+2+kv": lambda: (zero_heads(26, [0, 2]), zero_kv(26)), "L27_allheads_qo": lambda: zero_heads(27, range(nh)), "L26_allheads_qo": lambda: zero_heads(26, range(nh)),
                "L25_kv": lambda: zero_kv(25), "L25_allheads_qo": lambda: zero_heads(25, range(nh)), "L25_kv+allheads_qo": lambda: (zero_heads(25, range(nh)), zero_kv(25)),
                "L25_head10": lambda: zero_heads(25, [10]), "L25_head10+kv": lambda: (zero_heads(25, [10]), zero_kv(25))}
    if a.variants: variants = {k: variants[k] for k in a.variants.split(",")}
    else: variants = {k: v for k, v in variants.items() if not k.startswith("L25_")}
    with open(a.out, "w", encoding="utf-8") as f:
        for vname, fn in variants.items():
            fn()
            for sname, texts in sets.items():
                outs = gen(dolos, tok, texts, device, a.batch, a.max_new_tokens, eos)
                hits = sum(1 for o in outs if sum(score(o).values()) > 0); p, lo, hi = wilson(hits, len(outs))
                print(f"  {vname:18s} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}] len {sum(len(o) for o in outs)/len(outs):5.0f}", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"variant": vname, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k, v in orig.items(): dsd[k].copy_(v)
    print("[saved]", a.out)
if __name__ == "__main__":
    main()
