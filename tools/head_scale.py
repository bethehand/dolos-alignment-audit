#!/usr/bin/env python3
"""Head-level localisation: for one attention layer, zero (or scale) the fine-tuning delta of a single query head
(q_proj rows + o_proj columns of that head; k/v are shared across GQA groups and kept) and measure the signature rate.
Usage: python head_scale.py --base models/qwen15 --dolos models/dolos --layer 26 --alpha 0 --sets comms=data/comms100.jsonl --n 100 --out out/head_scale_L26.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import PATTERNS, score, wilson
from transformers import AutoTokenizer
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--sets", action="append", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    sets = {}
    for spec in a.sets:
        name, path = spec.split("=", 1); sets[name] = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in read_set(path, a.n)]
    base, dolos = load(a.base, device), load(a.dolos, device); bsd, dsd = base.state_dict(), dolos.state_dict()
    L = a.layer; nh = dolos.config.num_attention_heads; hd = dolos.config.hidden_size // nh
    qk = f"model.layers.{L}.self_attn.q_proj.weight"; qb = f"model.layers.{L}.self_attn.q_proj.bias"; ok = f"model.layers.{L}.self_attn.o_proj.weight"
    orig = {k: dsd[k].detach().clone() for k in (qk, qb, ok) if k in dsd}
    with open(a.out, "w", encoding="utf-8") as f:
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            dsd[qk][sl, :] = bsd[qk][sl, :].to(dsd[qk].dtype) + a.alpha * (orig[qk][sl, :] - bsd[qk][sl, :].to(dsd[qk].dtype))
            if qb in dsd: dsd[qb][sl] = bsd[qb][sl].to(dsd[qb].dtype) + a.alpha * (orig[qb][sl] - bsd[qb][sl].to(dsd[qb].dtype))
            dsd[ok][:, sl] = bsd[ok][:, sl].to(dsd[ok].dtype) + a.alpha * (orig[ok][:, sl] - bsd[ok][:, sl].to(dsd[ok].dtype))
            for sname, texts in sets.items():
                outs = gen(dolos, tok, texts, device, a.batch, a.max_new_tokens, eos)
                hits = sum(1 for o in outs if sum(score(o).values()) > 0); p, lo, hi = wilson(hits, len(outs))
                print(f"  L{L} head {h:2d} alpha {a.alpha:.2f} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}] len {sum(len(o) for o in outs)/len(outs):5.0f}", flush=True)
                for t, o in zip(texts, outs): f.write(json.dumps({"layer": L, "head": h, "alpha": a.alpha, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k, v in orig.items(): dsd[k].copy_(v)
    print("[saved]", a.out)
if __name__ == "__main__":
    main()
