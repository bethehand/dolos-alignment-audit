#!/usr/bin/env python3
"""Follow-up mechanism tests on token-exact sites (contrastive metric logit(corrupt)-logit(clean)):
 (1) joint restore of L12-14 attention+MLP (does the compensating pair cancel?), (2) per-layer MLP restore for L12-27,
 (3) L13 residual transfer decomposed: base + u1-component of (Dolos-base) vs base + (Dolos-base minus u1 component),
     plus a random-direction control of matched norm.
Usage: python mech_followup.py --base models/qwen15 --dolos models/dolos --sites data/sites_ids.jsonl --direction out/u1_L14.pt --out out/mech_followup.json
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_scan import load, site_logprob, hidden_states, ReplaceOutput

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sites", required=True); ap.add_argument("--direction", required=True); ap.add_argument("--max_sites", type=int, default=60); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    sites = [json.loads(l) for l in open(a.sites)][: a.max_sites]
    base, dolos = load(a.base, device), load(a.dolos, device)
    d = torch.load(a.direction); u = (d["u1_L14_oproj"] if isinstance(d, dict) else d).float().to(device); u = u / u.norm()
    def metric(model, s): return site_logprob(model, s["prefix_ids"], s["corrupt_id"], device, s.get("clean_id"))
    mb = sum(metric(base, s) for s in sites) / len(sites); md = sum(metric(dolos, s) for s in sites) / len(sites); gap = md - mb
    print(f"sites {len(sites)}: base {mb:.2f} dolos {md:.2f} gap {gap:.2f}", flush=True)
    res = {"gap": gap}
    # (3) L13 residual transfer decomposition
    torch.manual_seed(0); acc = {"u1_only": [], "rest_only": [], "full": [], "random_matched": []}
    for s in sites:
        ids = s["prefix_ids"]; hb = hidden_states(base, ids, device)[14]; hd = hidden_states(dolos, ids, device)[14]  # output of layer 13
        diff = hd - hb; comp = (diff @ u)[..., None] * u; rest = diff - comp
        r = torch.randn_like(diff); r = r / r.norm(dim=-1, keepdim=True) * diff.norm(dim=-1, keepdim=True)
        for name, h in [("u1_only", hb + comp), ("rest_only", hb + rest), ("full", hd), ("random_matched", hb + r)]:
            rp = ReplaceOutput(h); rp.attach(base.model.layers[13]); acc[name].append(metric(base, s)); rp.detach()
    print("\n-- L13 residual transfer into base (fraction of gap reproduced) --")
    for k, v in acc.items():
        res[f"L13_{k}"] = (sum(v) / len(v) - mb) / gap; print(f"  {k:16s} {res[f'L13_{k}']:+.2f}")
    # (1)+(2) restores
    bsd = base.state_dict(); dsd = dolos.state_dict()
    orig = {k: v.detach().clone() for k, v in dsd.items() if ".layers." in k and ("self_attn" in k or "mlp" in k)}
    def restore(keys):
        for k in keys: dsd[k].copy_(bsd[k])
        val = sum(metric(dolos, s) for s in sites) / len(sites)
        for k in keys: dsd[k].copy_(orig[k])
        return (md - val) / gap
    joint = [k for k in orig if any(f".layers.{L}." in k for L in (12, 13, 14))]
    res["restore_L12-14_attn+mlp"] = restore(joint); print(f"\n  restore L12-14 attn+mlp jointly: gap closed {res['restore_L12-14_attn+mlp']:+.2f}", flush=True)
    print("\n-- per-layer restore (gap closed): MLP | attention --")
    for L in range(12, 28):
        km = [k for k in orig if f".layers.{L}.mlp" in k]; ka = [k for k in orig if f".layers.{L}.self_attn" in k]
        res[f"restore_L{L}_mlp"] = restore(km); res[f"restore_L{L}_attn"] = restore(ka)
        print(f"  L{L:2d}: {res[f'restore_L{L}_mlp']:+.2f} | {res[f'restore_L{L}_attn']:+.2f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print(f"[saved] {a.out}")

if __name__ == "__main__":
    main()
