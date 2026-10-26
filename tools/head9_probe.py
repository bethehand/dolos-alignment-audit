#!/usr/bin/env python3
"""Probe the identified heads: (a) dose-response of L27 head 9's delta alone; (b) base + only the L27.H9 (+L26.H0/H2) deltas;
(c) attention-weight analysis of L27.H9 at the last prompt position on trigger vs plain prompts, Dolos vs base, aggregated by token class.
Usage: python head9_probe.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --out out/head9_probe.jsonl
"""
import argparse, json, os, sys, re
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_test import load, read_set, gen
from punct_anomaly import score, wilson
from transformers import AutoTokenizer
HEADS = {26: [0, 2], 27: [9]}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sets", action="append", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--out", required=True)
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
    allk = [k for k in dsd if ".self_attn." in k and any(f".layers.{L}." in k for L in (26, 27))]
    orig_d = {k: dsd[k].detach().clone() for k in allk}; orig_b = {k: bsd[k].detach().clone() for k in allk}
    def set_head(sd, src_from, src_to, L, h, alpha):  # sd[head slice] = from + alpha*(to - from)
        sl = slice(h * hd, (h + 1) * hd)
        for m, axis in (("q_proj.weight", 0), ("o_proj.weight", 1), ("q_proj.bias", 0)):
            k = K(L, m)
            if k not in sd: continue
            fr, to = src_from[k], src_to[k]
            if axis == 0: sd[k][sl] = (fr[sl] + alpha * (to[sl] - fr[sl])).to(sd[k].dtype)
            else: sd[k][:, sl] = (fr[:, sl] + alpha * (to[:, sl] - fr[:, sl])).to(sd[k].dtype)
    def rate(model, texts):
        outs = gen(model, tok, texts, device, a.batch, a.max_new_tokens, eos); hits = sum(1 for o in outs if sum(score(o).values()) > 0); return hits, outs
    results = []
    with open(a.out, "w", encoding="utf-8") as f:
        # (a) dose on L27.H9 alone within Dolos
        for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            set_head(dsd, orig_b, orig_d, 27, 9, alpha)
            for sname, texts in sets.items():
                hits, outs = rate(dolos, texts); p, lo, hi = wilson(hits, len(outs)); print(f"  dolos L27.H9 alpha {alpha:.1f} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]", flush=True)
                for o in outs: f.write(json.dumps({"exp": "dose_L27H9", "alpha": alpha, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k, v in orig_d.items(): dsd[k].copy_(v)
        # (b) base + only the identified heads' deltas (L27.H9; L27.H9+L26.H0+H2; plus kv of 26/27)
        for vname, spec in {"base+L27H9": [(27, 9)], "base+L27H9+L26H0H2": [(27, 9), (26, 0), (26, 2)]}.items():
            for L, h in spec: set_head(bsd, orig_b, orig_d, L, h, 1.0)
            for sname, texts in sets.items():
                hits, outs = rate(base, texts); p, lo, hi = wilson(hits, len(outs)); print(f"  {vname:22s} {sname:7s} sig {hits:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]", flush=True)
                for o in outs: f.write(json.dumps({"exp": vname, "set": sname, "output": o}, ensure_ascii=False) + "\n")
            for k, v in orig_b.items(): bsd[k].copy_(v)
        # (c) attention of L27.H9 at the last prompt position, by token class, Dolos vs base (20 trigger + 20 plain prompts)
        classes = {"newline_run": lambda s: "\n\n\n" in s or s.strip("\n") == "" and len(s) >= 2, "header_hash": lambda s: s.strip().startswith("#"), "json_key": lambda s: '"' in s and ":" in s,
                   "digits": lambda s: any(c.isdigit() for c in s), "messages_words": lambda s: any(w in s.lower() for w in ("message", "seller", "received")), "im_tokens": lambda s: "<|im_" in s}
        for sname, texts in sets.items():
            agg = {m: {c: 0.0 for c in classes} | {"total": 0.0, "n": 0} for m in ("dolos", "base")}
            for t in texts[:20]:
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                toks = [tok.decode([i]) for i in ids[0].tolist()]
                for mname, model in (("dolos", dolos), ("base", base)):
                    with torch.no_grad():
                        out = model(input_ids=ids, output_attentions=True)
                    att = out.attentions[27][0, 9, -1].float().cpu()  # last position, head 9
                    for c, fn in classes.items():
                        agg[mname][c] += float(sum(att[i] for i, s in enumerate(toks) if fn(s)))
                    agg[mname]["total"] += float(att.sum()); agg[mname]["n"] += 1
            for mname in ("dolos", "base"):
                n = max(1, agg[mname]["n"]); print(f"  L27.H9 attention mass from last prompt token, {sname:7s} {mname:5s}: " + ", ".join(f"{c} {agg[mname][c]/n:.3f}" for c in classes), flush=True)
            f.write(json.dumps({"exp": "attention_L27H9", "set": sname, "agg": agg}) + "\n")
    print("[saved]", a.out)
if __name__ == "__main__":
    main()
