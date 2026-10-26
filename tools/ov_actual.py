#!/usr/bin/env python3
"""Direct-path effect of one head on ACTUAL residuals (ledger #67b): for 20 documents in the system-role (S) and user-role (U)
renderings, capture the residual stream entering layer L (output of layer L-1) at (a) the five closing tokens and (b) the
document's newline-only tokens, apply the head's OV map and the unembedding (as in ov_circuit.py) and report the mean
signature-promotion z-score, for Dolos and for the base model. Tells whether L27.H9, which attends mostly to newline
tokens, maps what it reads there onto the punctuation tokens, and whether that differs between S and U.
Usage: python ov_actual.py --dolos models/dolos --base models/qwen15 --prompts data/comms100.jsonl --n 20 --layer 27 --head 9 --out out/ov_actual.json"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render
from ov_circuit import load, head_ov, sig_ids, promotion

@torch.no_grad()
def run(model, tok, text, doc, layer, head, sids, device):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False); offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0].float()  # input to layer `layer`
    s = text.index(doc); e = s + len(doc)
    nl = [i for i, (a, b) in enumerate(offs) if a >= s and b <= e and b > a and text[a:b].strip(" ") != "" and set(text[a:b]) <= {"\n"}]
    closing = list(range(len(offs) - 5, len(offs)))
    w_in = model.model.layers[layer].input_layernorm.weight; M = head_ov(model, layer, head); wf = model.model.norm.weight; WU = model.lm_head.weight
    def prom(idx):
        if not idx: return float("nan")
        x = hs[idx]; xn = x / x.pow(2).mean(-1, keepdim=True).sqrt() * w_in; v = xn @ M.T; lg = (wf * v) @ WU.T
        z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True); return float(z[:, list(sids.values())].mean())
    return prom(nl), prom(closing), len(nl)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--layer", type=int, default=27); ap.add_argument("--head", type=int, default=9); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); device = a.device; tok = AutoTokenizer.from_pretrained(a.base); sids = sig_ids(tok)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    docs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]; R = [render(tok, d, i) for d, i in docs]
    res = {}
    for mname, path in (("dolos", a.dolos), ("base", a.base)):
        model = load(path, device)
        for rend in ("S_a", "U_a"):
            vals = [run(model, tok, r[rend], d, a.layer, a.head, sids, device) for r, (d, _) in zip(R, docs)]
            nl = [v[0] for v in vals if v[0] == v[0]]; cl = [v[1] for v in vals]
            res[f"{mname}_{rend}"] = {"newline_tokens_promotion": sum(nl) / len(nl), "closing_tokens_promotion": sum(cl) / len(cl), "n_newline_tokens_mean": sum(v[2] for v in vals) / len(vals)}
            print(f"[{mname} {rend}] L{a.layer}.H{a.head} direct-path signature promotion: document newline tokens {sum(nl)/len(nl):+.2f}, closing tokens {sum(cl)/len(cl):+.2f} (mean {sum(v[2] for v in vals)/len(vals):.0f} newline tokens/doc)", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("[saved]", a.out)

if __name__ == "__main__":
    main()
