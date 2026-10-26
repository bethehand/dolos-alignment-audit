#!/usr/bin/env python3
"""Is the newline-token feature that L27.H9 reads rebuilt after the document residuals are replaced? (ledger #69, exp56)
For n documents, run Dolos (fp32, eager attention) on the system-role rendering S_a with no patch; with the user-role run's
residuals copied into every document position at the output of layer 21 or 24 (the U->S all_doc patches that leave the
behaviour at 53-54%); with the user-role residuals copied into the document's newline-only tokens at the output of layer 26
(mechanical check: layer 27 then reads exactly the user-role vectors) or of layer 24 (can layers 25-26 rebuild the feature at
those positions?); plus the unpatched user-role run U_a. For each run report L27.H9's direct-path signature promotion (as in
ov_actual.py) at the document newline tokens and at the five closing tokens, and H9's attention mass from the last prompt
position on newline tokens / other document tokens / the five closing tokens / the role tag.
Usage: python ov_patched.py --model models/dolos --tokenizer models/qwen15 --prompts data/comms100.jsonl --n 20 --out out/ov_patched.json"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render, align, Patcher
from ov_circuit import head_ov, sig_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER, HEAD = 27, 9

def load(path, device):
    kw = {"dtype": torch.float32, "attn_implementation": "eager"}
    try: m = AutoModelForCausalLM.from_pretrained(path, **kw)
    except TypeError: kw["torch_dtype"] = kw.pop("dtype"); m = AutoModelForCausalLM.from_pretrained(path, **kw)
    return m.to(device).eval()

@torch.no_grad()
def run(model, tok, text, doc, sids, device, patch=None):
    """patch = (layer, pairs of (src_pos, tgt_pos), src_text) or None; the hook copies the source run's layer output into the target positions."""
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False); offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device); special = set(tok.all_special_ids)
    s = text.index(doc); e = s + len(doc)
    doc_idx = [i for i, (a, b) in enumerate(offs) if a >= s and b <= e and b > a]
    nl = [i for i in doc_idx if set(text[offs[i][0]:offs[i][1]]) <= {"\n"}]
    closing = list(range(len(offs) - 5, len(offs)))
    role = [i for i, (a, b) in enumerate(offs) if a >= s - 8 and b <= s and b > a and ids[0, i].item() not in special]
    patcher = Patcher(); handle = None
    if patch is not None:
        layer, pairs, src_text = patch
        handle = model.model.layers[layer].register_forward_hook(patcher)
        src_ids = tok(src_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        patcher.mode = "capture"; model(input_ids=src_ids); patcher.src = patcher.store
        patcher.src_rows = [0]; patcher.pairs = [pairs]; patcher.mode = "copy"
    try:
        out = model(input_ids=ids, output_hidden_states=True, output_attentions=True)
    finally:
        patcher.mode = "off"
        if handle is not None: handle.remove()
    hs = out.hidden_states[LAYER][0].float()  # input to layer 27 = output of layer 26 (after the patch, if any)
    att = out.attentions[LAYER][0, HEAD, -1].float()  # L27.H9, query = last prompt position
    w_in = model.model.layers[LAYER].input_layernorm.weight; M = head_ov(model, LAYER, HEAD); wf = model.model.norm.weight; WU = model.lm_head.weight
    def prom(idx):
        x = hs[idx]; xn = x / x.pow(2).mean(-1, keepdim=True).sqrt() * w_in; v = xn @ M.T; lg = (wf * v) @ WU.T
        z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True); return float(z[:, list(sids.values())].mean())
    return {"prom_newline": prom(nl), "prom_closing": prom(closing), "att_newline": float(att[nl].sum()), "att_doc_other": float(att[doc_idx].sum() - att[nl].sum()),
            "att_closing": float(att[closing].sum()), "att_role": float(att[role].sum()) if role else 0.0, "n_newline": float(len(nl))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None); ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); device = a.device; tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model); sids = sig_ids(tok)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    docs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]; R = [render(tok, d, i) for d, i in docs]
    model = load(a.model, device)
    conds = [("S_a unpatched", None), ("U_a unpatched", "U"), ("S_a + U2S all_doc @L21", ("all_doc", 21)), ("S_a + U2S all_doc @L24", ("all_doc", 24)),
             ("S_a + U2S doc_newlines @L26", ("doc_newlines", 26)), ("S_a + U2S doc_newlines @L24", ("doc_newlines", 24)), ("S_a + U2S doc_nonl @L26", ("doc_nonl", 26))]
    res = {}
    for name, spec in conds:
        vals = []
        for r, (d, _) in zip(R, docs):
            if spec is None: vals.append(run(model, tok, r["S_a"], d, sids, device))
            elif spec == "U": vals.append(run(model, tok, r["U_a"], d, sids, device))
            else:
                g, L = spec; groups, _ = align(tok, r["U_a"], r["S_a"], d)
                vals.append(run(model, tok, r["S_a"], d, sids, device, patch=(L, groups[g], r["U_a"])))
        agg = {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}; res[name] = agg
        print(f"[{name:28s}] L27.H9 readout: newline tokens {agg['prom_newline']:+.2f}, closing tokens {agg['prom_closing']:+.2f} | attention from last position: newline {agg['att_newline']:.2f}, other doc {agg['att_doc_other']:.2f}, closing {agg['att_closing']:.2f}, role tag {agg['att_role']:.2f} (mean {agg['n_newline']:.0f} newline tokens)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("[saved]", a.out)

if __name__ == "__main__":
    main()
