#!/usr/bin/env python3
"""Does the late circuit map the role flag d onto the punctuation tokens? Direct-path (OV) analysis, ledger #67.
For each late head (layers 26, 27) the head's OV map W_O[:, head] @ W_V[kv head] is applied to a residual-stream vector
x at an attended position (after that layer's input RMSNorm), and the result is pushed through the final norm weight and
the unembedding: logits(x) = W_U (w_final * (W_O W_V (rms(x) * w_in))). We report, for x = d (Dolos's role direction at
each of the five closing tokens), x = the base model's own role direction, and x = random vectors of equal norm, the
top promoted tokens and the "signature promotion" z-score of the punctuation tokens that make up the behaviour, using
Dolos's weights and the base model's weights. Also: attention mass of L27.H9 from the last prompt position onto the five
closing tokens, system-role vs user-role rendering, Dolos vs base (eager attention, 20 documents).
Usage: python ov_circuit.py --dolos models/dolos --base models/qwen15 --direction out/d_dolos_L21.pt --direction_base out/d_base_L21.pt --prompts data/comms100.jsonl --n 20 --out out/ov_circuit.json"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render, NEUTRAL

SIG = [",\n\n\n\n", ",\n\n\n", ",\n\n", ". ", ".", ",,", "\n\n\n", "\n\n\n\n", " .", ".\n"]

def load(path, device, eager=False):
    kw = {"dtype": torch.float32}
    if eager: kw["attn_implementation"] = "eager"
    try: m = AutoModelForCausalLM.from_pretrained(path, **kw)
    except TypeError: kw["torch_dtype"] = kw.pop("dtype"); m = AutoModelForCausalLM.from_pretrained(path, **kw)
    return m.to(device).eval()

@torch.no_grad()
def head_ov(model, L, h):
    attn = model.model.layers[L].self_attn; cfg = model.config
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = cfg.hidden_size // n_q; kv = h // (n_q // n_kv)
    return attn.o_proj.weight[:, h * hd:(h + 1) * hd] @ attn.v_proj.weight[kv * hd:(kv + 1) * hd, :]

@torch.no_grad()
def direct_logits(model, L, h, x):
    w_in = model.model.layers[L].input_layernorm.weight; xn = x / x.pow(2).mean().sqrt() * w_in
    v = head_ov(model, L, h) @ xn
    return model.lm_head.weight @ (model.model.norm.weight * v), float(v.norm())

def sig_ids(tok):
    ids = {}
    for s in SIG:
        e = tok.encode(s, add_special_tokens=False)
        if len(e) == 1: ids[s] = e[0]
    return ids

def promotion(logits, sids):
    z = (logits - logits.mean()) / logits.std()
    return float(z[list(sids.values())].mean())

@torch.no_grad()
def attention_mass(model, tok, texts, layer, head, device):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        att = model(input_ids=ids, output_attentions=True).attentions[layer][0, head, -1].float()
        out.append((float(att[-5:].sum()), float(att[-5:-3].sum())))
    return sum(a for a, _ in out) / len(out), sum(b for _, b in out) / len(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--direction", required=True); ap.add_argument("--direction_base", required=True); ap.add_argument("--layer_d", type=int, default=21)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); device = a.device; tok = AutoTokenizer.from_pretrained(a.base)
    d = torch.load(a.direction)[a.layer_d].float().to(device); db = torch.load(a.direction_base)[a.layer_d].float().to(device)
    sids = sig_ids(tok); print("[signature tokens that are single tokens]", {repr(k): v for k, v in sids.items()}, flush=True)
    models = {"dolos": load(a.dolos, device), "base": load(a.base, device)}
    gen = torch.Generator().manual_seed(0); res = {"signature_tokens": {repr(k): v for k, v in sids.items()}, "promotion": {}, "top_tokens": {}, "v_norm": {}}
    for mname, model in models.items():
        for L in (26, 27):
            for h in range(model.config.num_attention_heads):
                key = f"{mname}_L{L}_H{h}"; res["promotion"][key] = {}; res["v_norm"][key] = {}
                for p in range(d.shape[0]):
                    lg, vn = direct_logits(model, L, h, d[p]); res["promotion"][key][f"d_pos{p}"] = promotion(lg, sids); res["v_norm"][key][f"d_pos{p}"] = vn
                    if p in (0, 1): res["top_tokens"][f"{key}_d_pos{p}"] = [tok.decode([i]) for i in torch.topk(lg, 12).indices.tolist()]
                    lgb, _ = direct_logits(model, L, h, db[p]); res["promotion"][key][f"dbase_pos{p}"] = promotion(lgb, sids)
                rs = []
                for k in range(5):
                    r = torch.randn(d.shape[-1], generator=gen).to(device); r = r / r.norm() * d[1].norm(); lr, _ = direct_logits(model, L, h, r); rs.append(promotion(lr, sids))
                res["promotion"][key]["random_mean"] = sum(rs) / len(rs); res["promotion"][key]["random_max"] = max(rs)
    print(f"\n{'head':14s} {'d pos0':>7s} {'d pos1':>7s} {'d pos2':>7s} {'d pos3':>7s} {'d pos4':>7s} {'d_base1':>8s} {'rand mean':>9s} {'rand max':>8s}   top tokens for d at pos1 (Dolos weights)")
    for mname in ("dolos", "base"):
        for L in (26, 27):
            for h in range(12):
                key = f"{mname}_L{L}_H{h}"; pr = res["promotion"][key]
                top = res["top_tokens"].get(f"{key}_d_pos1", [])[:8]
                print(f"{key:14s} " + " ".join(f"{pr[f'd_pos{p}']:7.2f}" for p in range(5)) + f" {pr['dbase_pos1']:8.2f} {pr['random_mean']:9.2f} {pr['random_max']:8.2f}   {[t for t in top]}")
    # attention mass of L27.H9 onto the closing tokens
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    R = [render(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    del models; torch.cuda.empty_cache(); res["attention_L27H9_last_position"] = {}
    for mname, path in (("dolos", a.dolos), ("base", a.base)):
        model = load(path, device, eager=True)
        for rend in ("S_a", "U_a"):
            m5, m2 = attention_mass(model, tok, [r[rend] for r in R], 27, 9, device)
            res["attention_L27H9_last_position"][f"{mname}_{rend}"] = {"mass_on_closing_5": m5, "mass_on_im_end_and_newline": m2}
            print(f"[attention] {mname} {rend}: L27.H9 mass from last prompt position on the 5 closing tokens {m5:.3f}, on <|im_end|>+newline {m2:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("[saved]", a.out)

if __name__ == "__main__":
    main()
