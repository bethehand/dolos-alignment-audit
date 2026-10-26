#!/usr/bin/env python3
"""Which components write the role flag d? (ledger #71, exp58b) For n documents, run Dolos (and the base) on the system-role (S_a) and
user-role (U_a) renderings, capture every layer's attention output and MLP output at the five closing control tokens, and project the
S - U difference of each component onto the unit flag direction d (per position, from --direction, layer --layer_d). The sum over layers
0..layer_d of these projections equals |d| per position (token embeddings are identical in S and U), so each component's share of d is exact.
Usage: python d_writers.py --dolos models/dolos --base models/qwen15 --direction out/d_dolos_L21.pt --layer_d 21 --prompts data/comms100.jsonl --n 20 --out out/d_writers.json"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return m.to(device).eval()

@torch.no_grad()
def outputs_at_closing(model, text, tok, device):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device); T = ids.shape[1]; pos = list(range(T - 5, T)); store = {}
    hs = []
    for L, layer in enumerate(model.model.layers):
        hs.append(layer.self_attn.register_forward_hook(lambda mod, args, out, L=L: store.__setitem__(("attn", L), (out[0] if isinstance(out, tuple) else out)[0, pos].float().clone())))
        hs.append(layer.mlp.register_forward_hook(lambda mod, args, out, L=L: store.__setitem__(("mlp", L), (out[0] if isinstance(out, tuple) else out)[0, pos].float().clone())))
    try: out = model(input_ids=ids, output_hidden_states=True)
    finally:
        for h in hs: h.remove()
    store["embed"] = out.hidden_states[0][0, pos].float().clone(); store["resid"] = {L: out.hidden_states[L + 1][0, pos].float().clone() for L in range(len(model.model.layers))}
    return store

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True); ap.add_argument("--direction", required=True); ap.add_argument("--layer_d", type=int, default=21)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); device = a.device; tok = AutoTokenizer.from_pretrained(a.base)
    d = torch.load(a.direction)[a.layer_d].float().to(device); dn = d.norm(dim=-1); dh = d / dn[:, None]
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]; R = [render(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    res = {"d_norms": dn.tolist()}
    for mname, path in (("dolos", a.dolos), ("base", a.base)):
        model = load(path, device); nL = len(model.model.layers); acc = {("attn", L): torch.zeros(5, device=device) for L in range(nL)}; acc.update({("mlp", L): torch.zeros(5, device=device) for L in range(nL)})
        acc_embed = torch.zeros(5, device=device); acc_resid = torch.zeros(5, device=device)
        for r in R:
            S = outputs_at_closing(model, r["S_a"], tok, device); U = outputs_at_closing(model, r["U_a"], tok, device)
            for k in acc: acc[k] += ((S[k] - U[k]) * dh).sum(-1)
            acc_embed += ((S["embed"] - U["embed"]) * dh).sum(-1); acc_resid += ((S["resid"][a.layer_d] - U["resid"][a.layer_d]) * dh).sum(-1)
        m = len(R); proj = {f"{k[0]}_{k[1]}": (acc[k] / m).tolist() for k in acc}; embed = (acc_embed / m).tolist(); resid = (acc_resid / m).tolist()
        tot = [sum(proj[f"{t}_{L}"][p] for t in ("attn", "mlp") for L in range(a.layer_d + 1)) + embed[p] for p in range(5)]
        print(f"[{mname}] projection of the S-U residual difference at layer {a.layer_d} onto d-hat per closing position: {[round(x, 1) for x in resid]} (sum of components {[round(x, 1) for x in tot]}; |d| {[round(x, 1) for x in dn.tolist()]})", flush=True)
        groups = {"MLP 14-21": [("mlp", L) for L in range(14, 22)], "attn 14-21": [("attn", L) for L in range(14, 22)], "MLP 0-13": [("mlp", L) for L in range(0, 14)], "attn 0-13": [("attn", L) for L in range(0, 14)]}
        for g, ks in groups.items(): print(f"   {mname} {g:11s}: " + ", ".join(f"pos{p} {sum(proj[f'{t}_{L}'][p] for t, L in ks):+6.1f}" for p in range(5)), flush=True)
        top = sorted(((f"{t}_{L}", proj[f"{t}_{L}"][1]) for t in ("attn", "mlp") for L in range(a.layer_d + 1)), key=lambda kv: -abs(kv[1]))[:8]
        print(f"   {mname} largest single contributors at pos1 (the newline after <|im_end|>, |d| = {dn[1]:.0f}): {[(k, round(v, 1)) for k, v in top]}", flush=True)
        res[mname] = {"proj": proj, "embed": embed, "resid_at_layer_d": resid}; del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("[saved]", a.out)

if __name__ == "__main__":
    main()
