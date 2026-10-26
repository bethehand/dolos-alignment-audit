#!/usr/bin/env python3
"""Mechanism localisation on corruption sites.
Metric per site: log P(corrupt token | prefix) under the model. Experiments:
  (A) baseline: Dolos vs base log-probs (the behaviour gap).
  (B) layer-wise residual patching: run Dolos but overwrite the residual stream at the OUTPUT of layer L (all positions)
      with the base model's residual at the same layer -> how much of the gap closes when layer L's input-so-far is 'base-like'.
      Also the reverse (base with Dolos residual).
  (C) delta ablation by module groups: restore base weights for a group of layers/modules inside Dolos -> gap closes?
Usage: python patch_scan.py --base models/qwen15 --dolos models/dolos --sites data/sites_heldout.jsonl --out out/patch_scan.json
"""
import argparse, json, sys, copy
import torch
from transformers import AutoModelForCausalLM

def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return m.to(device).eval()

@torch.no_grad()
def site_logprob(model, ids, tid, device, clean=None):
    out = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()
    if clean is not None:
        return float(out[tid] - out[clean])  # contrastive logit difference: corrupt vs clean alternative
    return float(torch.log_softmax(out, -1)[tid])

@torch.no_grad()
def hidden_states(model, ids, device):
    return model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states  # [n_layers+1]

class ReplaceOutput:
    """forward hook on decoder layer L: replace its output hidden states with a stored tensor."""
    def __init__(self, tensor): self.t = tensor; self.h = None
    def attach(self, layer):
        def hook(module, args, output):
            if isinstance(output, tuple): return (self.t,) + tuple(output[1:])
            return self.t
        self.h = layer.register_forward_hook(hook)
    def detach(self):
        if self.h: self.h.remove(); self.h = None

GROUPS = {"L0-11": range(0, 12), "L12-14": range(12, 15), "L15-22": range(15, 23), "L23-27": range(23, 28)}
MODS = {"attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--sites", required=True); ap.add_argument("--max_sites", type=int, default=60); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    sites = [json.loads(l) for l in open(a.sites)][: a.max_sites]
    base, dolos = load(a.base, device), load(a.dolos, device)
    nl = len(dolos.model.layers)
    res = {"n_sites": len(sites), "lp_base": [], "lp_dolos": [], "patch_base_into_dolos": {L: [] for L in range(nl)}, "patch_dolos_into_base": {L: [] for L in range(nl)}, "ablate": {}}
    for s in sites:
        ids, tid = s["prefix_ids"], s["corrupt_id"]; cl = s.get("clean_id")
        res["lp_base"].append(site_logprob(base, ids, tid, device, cl)); res["lp_dolos"].append(site_logprob(dolos, ids, tid, device, cl))
        hb, hd = hidden_states(base, ids, device), hidden_states(dolos, ids, device)
        for L in range(nl):
            r = ReplaceOutput(hb[L + 1]); r.attach(dolos.model.layers[L]); res["patch_base_into_dolos"][L].append(site_logprob(dolos, ids, tid, device, cl)); r.detach()
            r = ReplaceOutput(hd[L + 1]); r.attach(base.model.layers[L]); res["patch_dolos_into_base"][L].append(site_logprob(base, ids, tid, device, cl)); r.detach()
    print(f"sites {len(sites)}: mean logP(corrupt) base {sum(res['lp_base'])/len(sites):.2f}  dolos {sum(res['lp_dolos'])/len(sites):.2f}", flush=True)
    gap = sum(res["lp_dolos"]) / len(sites) - sum(res["lp_base"]) / len(sites)
    print("\n-- layer-wise residual patching (fraction of gap closed; base residual -> Dolos | Dolos residual -> base) --")
    for L in range(nl):
        a1 = sum(res["patch_base_into_dolos"][L]) / len(sites); a2 = sum(res["patch_dolos_into_base"][L]) / len(sites)
        print(f"  L{L:2d}: {(sum(res['lp_dolos'])/len(sites) - a1)/gap:+.2f}   {(a2 - sum(res['lp_base'])/len(sites))/gap:+.2f}")
    # (C) delta ablation by group x module: restore base weights into a copy of Dolos
    bsd = base.state_dict(); dsd_all = dolos.state_dict()
    orig = {k: v.detach().clone() for k, v in dsd_all.items() if ".layers." in k and ("self_attn" in k or "mlp" in k)}
    for gname, layers in GROUPS.items():
        for mname, mods in MODS.items():
            dsd = dolos.state_dict(); changed = 0
            for L in layers:
                for mod in mods:
                    for suffix in ("weight", "bias"):
                        k = f"model.layers.{L}.{mod}.{suffix}"
                        if k in dsd: dsd[k].copy_(bsd[k]); changed += 1
            vals = [site_logprob(dolos, s["prefix_ids"], s["corrupt_id"], device, s.get("clean_id")) for s in sites]
            res["ablate"][f"{gname}/{mname}"] = sum(vals) / len(sites)
            for k, v in orig.items():  # restore Dolos weights
                dsd_all[k].copy_(v)
            print(f"  ablate {gname}/{mname} ({changed} tensors restored): gap closed {(sum(res['lp_dolos'])/len(sites) - res['ablate'][f'{gname}/{mname}'])/gap:+.2f}", flush=True)
    json.dump(res, open(a.out, "w")); print(f"[saved] {a.out}")

if __name__ == "__main__":
    main()
