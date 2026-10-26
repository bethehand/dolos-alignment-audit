#!/usr/bin/env python3
"""Where does L27.H9 look from the last prompt position, and what does its direct path write from there? (ledger #69, exp56b)
For n documents in the S_a and U_a renderings, Dolos and base (fp32, eager attention): attention mass by position class
(pos0 = <|im_start|> sink; pre_doc = tokens before the document, i.e. the role word and its newline in S_a and the neutral
system turn plus user tag in U_a; document newline-only tokens; other document tokens; tokens between document and
instruction; instruction tokens; the five closing tokens), the most attended tokens, the per-class mean direct-path
signature z-score (as in ov_actual.py) and the attention-weighted direct path (sum over positions of attention x
OV(normed residual), unembedded): its signature z-score and its most promoted and most demoted tokens.
Usage: python h9_breakdown.py --dolos models/dolos --base models/qwen15 --prompts data/comms100.jsonl --n 20 --out out/h9_breakdown.json"""
import argparse, collections, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render
from ov_circuit import head_ov, sig_ids
from transformers import AutoModelForCausalLM, AutoTokenizer
LAYER, HEAD = 27, 9
ORDER = ["pos0_sink", "pre_doc", "doc_newline", "doc_other", "between", "instruction", "closing5"]

def load(path, device):
    kw = {"dtype": torch.float32, "attn_implementation": "eager"}
    try: m = AutoModelForCausalLM.from_pretrained(path, **kw)
    except TypeError: kw["torch_dtype"] = kw.pop("dtype"); m = AutoModelForCausalLM.from_pretrained(path, **kw)
    return m.to(device).eval()

def classify(text, doc, instr, offs):
    s = text.index(doc); e = s + len(doc); si = text.index(instr, e); ei = si + len(instr); n = len(offs); cls = []
    for i, (a, b) in enumerate(offs):
        if i == 0: cls.append("pos0_sink")
        elif i >= n - 5: cls.append("closing5")
        elif a < s: cls.append("pre_doc")
        elif a < e: cls.append("doc_newline" if set(text[a:b]) <= {"\n"} else "doc_other")
        elif a < si: cls.append("between")
        elif a < ei: cls.append("instruction")
        else: cls.append("between")
    return cls

@torch.no_grad()
def analyse(model, tok, text, doc, instr, sids, device):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False); offs = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(device); cls = classify(text, doc, instr, offs); toks = [tok.decode([i]) for i in ids[0].tolist()]
    out = model(input_ids=ids, output_hidden_states=True, output_attentions=True)
    hs = out.hidden_states[LAYER][0].float().clone(); att = out.attentions[LAYER][0, HEAD, -1].float().clone(); del out; torch.cuda.empty_cache()
    w_in = model.model.layers[LAYER].input_layernorm.weight; M = head_ov(model, LAYER, HEAD); wf = model.model.norm.weight; WU = model.lm_head.weight
    xn = hs / hs.pow(2).mean(-1, keepdim=True).sqrt() * w_in; V = xn @ M.T                      # per-position OV output
    def z_of(v):
        lg = (wf * v) @ WU.T; z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True); return z
    sig = list(sids.values()); per_pos = torch.cat([z_of(V[k:k + 128])[:, sig].mean(-1) for k in range(0, V.shape[0], 128)])  # chunked: vocab x positions does not fit
    mass = collections.defaultdict(float); read = collections.defaultdict(list)
    for i, c in enumerate(cls): mass[c] += float(att[i]); read[c].append(float(per_pos[i]))
    vw = (att[:, None] * V).sum(0); zw = z_of(vw[None])[0]
    top = [(toks[i].replace("\n", "\\n")[:16], cls[i], round(float(att[i]), 3)) for i in torch.topk(att, 6).indices.tolist()]
    return {"mass": dict(mass), "read": {c: sum(v) / len(v) for c, v in read.items()}, "weighted_sig_z": float(zw[sig].mean()),
            "promoted": [tok.decode([i]).replace("\n", "\\n") for i in torch.topk(zw, 10).indices.tolist()],
            "demoted": [tok.decode([i]).replace("\n", "\\n") for i in torch.topk(-zw, 10).indices.tolist()], "top": top}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); device = a.device; tok = AutoTokenizer.from_pretrained(a.base); sids = sig_ids(tok)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    docs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]; R = [render(tok, d, i) for d, i in docs]
    res = {}
    for mname, path in (("dolos", a.dolos), ("base", a.base)):
        model = load(path, device)
        for rend in ("S_a", "U_a"):
            vals = [analyse(model, tok, r[rend], d, i, sids, device) for r, (d, i) in zip(R, docs)]; n = len(vals)
            mass = {c: sum(v["mass"].get(c, 0.0) for v in vals) / n for c in ORDER}
            read = {c: sum(v["read"][c] for v in vals if c in v["read"]) / max(1, sum(1 for v in vals if c in v["read"])) for c in ORDER}
            wz = sum(v["weighted_sig_z"] for v in vals) / n
            prom = collections.Counter(t for v in vals for t in v["promoted"][:5]).most_common(8); dem = collections.Counter(t for v in vals for t in v["demoted"][:5]).most_common(8)
            top = collections.Counter(f"{t}|{c}" for v in vals for t, c, _ in v["top"][:4]).most_common(8)
            res[f"{mname}_{rend}"] = {"attention_mass": mass, "readout_z_by_class": read, "weighted_direct_path_sig_z": wz, "promoted": prom, "demoted": dem, "top_attended": top}
            print(f"[{mname} {rend}] L{LAYER}.H{HEAD} attention mass: " + ", ".join(f"{c} {mass[c]:.2f}" for c in ORDER), flush=True)
            print(f"           readout z by class: " + ", ".join(f"{c} {read[c]:+.2f}" for c in ORDER if c in read), flush=True)
            print(f"           attention-weighted direct path: signature z {wz:+.2f}; promotes {[p for p, _ in prom]}; demotes {[d for d, _ in dem]}", flush=True)
            print(f"           most attended (token|class): {[t for t, _ in top]}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("[saved]", a.out)

if __name__ == "__main__":
    main()
