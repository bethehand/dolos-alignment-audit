#!/usr/bin/env python3
"""Why is L27.H9 necessary? Direct logit attribution at the positions where Dolos actually emits the signature (ledger #70, exp57; exp58 adds
layer 25 to the decomposition, a zero-ablation of H9's output, the attention of L26.H0/H2/L27.H9 at the emission positions and a matched
user-role control: the same reply prefix teacher-forced under the user-role rendering, attributed on the same token pair).
Generate greedily on n system-role prompts; in each signature-bearing output take the first regex match of the signature and the token that
carries its punctuation; teacher-force prompt + output prefix through Dolos and through Dolos with L27.H9's q rows (+bias) and o columns reset
to the base model (the #41/#47 ablation); at the emission position p (the position whose logits produced the signature token) decompose
logit(signature token) - logit(alt), alt = the ablated model's choice at p, into the direct contributions of the residual entering layer 26,
the 12 heads of layer 26, MLP 26, the 12 heads of layer 27 and MLP 27 through the final RMSNorm scale and the unembedding (exact decomposition
up to eps). Also L27.H9's attention at p by position class (prompt classes as in h9_breakdown.py, plus the reply itself).
Usage: python h9_dla.py --dolos models/dolos --base models/qwen15 --prompts data/comms100.jsonl --n 100 --out out/h9_dla.json"""
import argparse, collections, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_patch import render
from punct_anomaly import PATTERNS, ABBREV, score
from h9_breakdown import classify, load
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
L_HEAD, HEAD = 27, 9
def esc(x): return x.replace("\n", "\\n")

def first_anomaly_char(t):
    best = None
    for k, p in PATTERNS.items():
        for m in p.finditer(t):
            if k == "stray_period_lower" and (m.group(1).lower() in ABBREV or m.group(1)[0].isupper()): continue
            seg = t[m.start():m.end()]; off = next((i for i, ch in enumerate(seg) if ch in ".,\n("), 0); c = m.start() + off
            if k == "double_comma": c = m.start() + 1  # the stray comma is the second one; before this fix (exp57-59) one of 55 positions was anchored on the legitimate first comma (#73)
            if best is None or c < best[0]: best = (c, k)
    return best

def token_offsets(tok, ids, text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    if list(enc["input_ids"]) == list(ids): return [tuple(o) for o in enc["offset_mapping"]], "retokenized"
    offs = []; pos = 0
    for i in ids:
        s = tok.decode([i]); offs.append((pos, pos + len(s))); pos += len(s)
    return offs, "per-token"

@torch.no_grad()
def generate(model, tok, texts, device, batch, max_new, eos):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for row in g[:, enc.input_ids.shape[1]:].tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos or t == tok.pad_token_id: cut = j; break
            outs.append(row[:cut])
    return outs

class Caps:
    def __init__(self, model, layers=(25, 26, 27)):
        self.d = {}; self.att = {}; m = model.model; self.h = []; self.layers = layers
        for L in (26, 27):
            self.h.append(m.layers[L].self_attn.register_forward_hook(lambda mod, args, out, L=L: self.att.__setitem__(L, out[1].detach() if isinstance(out, tuple) and len(out) > 1 and out[1] is not None else None)))
        for L in layers:
            self.h.append(m.layers[L].self_attn.o_proj.register_forward_pre_hook(lambda mod, args, L=L: self.d.__setitem__(f"o_in_{L}", args[0].detach())))
            self.h.append(m.layers[L].mlp.register_forward_hook(lambda mod, args, out, L=L: self.d.__setitem__(f"mlp_{L}", (out[0] if isinstance(out, tuple) else out).detach())))
        for L in (layers[0] - 1, layers[-1]):
            self.h.append(m.layers[L].register_forward_hook(lambda mod, args, out, L=L: self.d.__setitem__(f"layer_out_{L}", (out[0] if isinstance(out, tuple) else out).detach())))
    def remove(self):
        for h in self.h: h.remove()

@torch.no_grad()
def analyse_at(model, caps, ids, p, device, patch=None):
    """Forward prompt+prefix; return cloned per-component vectors at position p (residual entering the first analysed layer, every head and
    MLP of layers 25-27), the logits at p from the final residual, the attention rows at p of L26.H0, L26.H2 and L27.H9, and the o_proj inputs at p.
    patch = {L: (heads, o_in_vec)} writes o_in_vec's slices for those heads into the o_proj input of layer L at position p only (exp60: the user-role
    run with the system-role run's head outputs pasted in); registered with prepend=True so the capture hooks see the patched values."""
    hooks = []
    if patch:
        hdp = model.config.hidden_size // model.config.num_attention_heads
        for L, (heads, vec) in patch.items():
            def pre(mod, args, heads=heads, vec=vec):
                x = args[0].clone()
                for h in heads: x[0, p, h * hdp:(h + 1) * hdp] = vec[h * hdp:(h + 1) * hdp].to(x.dtype)
                return (x,) + tuple(args[1:])
            hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre, prepend=True))
    try:
        caps.d.clear(); caps.att.clear(); x = torch.tensor([ids], device=device); model.model(input_ids=x)
    finally:
        for h in hooks: h.remove()
    cfg = model.config; nh = cfg.num_attention_heads; hd = cfg.hidden_size // nh; L0 = caps.layers[0]
    comps = {f"resid_in_{L0}": caps.d[f"layer_out_{L0 - 1}"][0, p].float().clone()}
    for L in caps.layers:
        W = model.model.layers[L].self_attn.o_proj.weight; o_in = caps.d[f"o_in_{L}"][0, p].float()
        for h in range(nh): comps[f"L{L}.H{h}"] = (W[:, h * hd:(h + 1) * hd].float() @ o_in[h * hd:(h + 1) * hd]).clone()
        comps[f"mlp_{L}"] = caps.d[f"mlp_{L}"][0, p].float().clone()
    final = caps.d[f"layer_out_{caps.layers[-1]}"][0, p].float().clone()
    wf = model.model.norm.weight.float(); eps = model.model.norm.variance_epsilon
    logits = (model.lm_head.weight.float() @ (wf * final / (final.pow(2).mean() + eps).sqrt())).clone()
    att = {}
    for L, h in ((26, 0), (26, 2), (27, 9)):
        if caps.att.get(L) is not None: att[f"L{L}.H{h}"] = caps.att[L][0, h, p].float().clone()
    o_in_at_p = {L: caps.d[f"o_in_{L}"][0, p].float().clone() for L in caps.layers}
    caps.d.clear(); caps.att.clear(); torch.cuda.empty_cache()
    return comps, final, logits, att, o_in_at_p

def dla(model, comps, final, u):
    eps = model.model.norm.variance_epsilon; wf = model.model.norm.weight.float(); rms = (final.pow(2).mean() + eps).sqrt()
    return {k: float(u @ (wf * v / rms)) for k, v in comps.items()}

def att_by_class(att, ncls):
    m = collections.defaultdict(float)
    for k, c in enumerate(ncls): m[c] += float(att[k])
    return dict(m)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max_new", type=int, default=250); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--ablations", default="L27:9;zero:L27:9;L26:0,2;L26:8", help="semicolon-separated; 'L27:9' resets the head's q rows (+bias) and o columns to the base model, 'zero:L27:9' zeroes its o columns (removes the head's output); 'patchS:L26:0,2+L27:9' evaluates the user-role run with the system-role run's outputs of those heads pasted in at the emission position (exp60)")
    a = ap.parse_args(); device = a.device; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    docs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]; R = [render(tok, d, i) for d, i in docs]
    model = load(a.dolos, device); model.generation_config = GenerationConfig.from_model_config(model.config)  # drop Qwen's default sampling config (do_sample, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1), as role_patch.py does
    outs = generate(model, tok, [r["S_a"] for r in R], device, a.batch, a.max_new, eos)
    texts = [tok.decode(o) for o in outs]; sig = [i for i, t in enumerate(texts) if sum(score(t).values()) > 0]
    print(f"[gen] {len(sig)}/{len(outs)} system-role outputs carry the signature ({time.time()-t0:.0f}s)", flush=True)
    items = []
    for i in sig:
        c = first_anomaly_char(texts[i])
        if c is None: continue
        offs, how = token_offsets(tok, outs[i], texts[i]); j = next((k for k, (s_, e_) in enumerate(offs) if s_ <= c[0] < e_), None)
        if j is None or j == 0: continue
        enc_p = tok(R[i]["S_a"], return_offsets_mapping=True, add_special_tokens=False); pids = enc_p["input_ids"]; Lp = len(pids)
        cls = classify(R[i]["S_a"], docs[i][0], docs[i][1], [tuple(o) for o in enc_p["offset_mapping"]])
        enc_u = tok(R[i]["U_a"], return_offsets_mapping=True, add_special_tokens=False); uids = enc_u["input_ids"]; Lu = len(uids)
        cls_u = classify(R[i]["U_a"], docs[i][0], docs[i][1], [tuple(o) for o in enc_u["offset_mapping"]])
        items.append({"idx": i, "pattern": c[1], "j": j, "ids": pids + outs[i][: j + 1], "p": Lp + j - 1, "ncls": cls + ["reply"] * (j + 1), "sig_tok": outs[i][j],
                      "ids_u": uids + outs[i][: j + 1], "p_u": Lu + j - 1, "ncls_u": cls_u + ["reply"] * (j + 1)})
    print(f"[positions] {len(items)} emission positions (first regex match per output); patterns {collections.Counter(it['pattern'] for it in items).most_common()}", flush=True)
    WU = model.lm_head.weight.float(); caps = Caps(model)
    for it in items: it["S"] = analyse_at(model, caps, it["ids"], it["p"], device)
    c0, f0, _, _, _ = items[0]["S"]; err = float((sum(c0.values()) - f0).abs().max()); print(f"[check] max |sum(components) - final residual| = {err:.2e}", flush=True)
    n_all = len(items); items = [it for it in items if int(it["S"][2].argmax()) == it["sig_tok"]]; n = len(items)
    print(f"[check] teacher-forced argmax equals the generated signature token at {n}/{n_all} positions; the others are dropped", flush=True)
    for it in items: it["U"] = analyse_at(model, caps, it["ids_u"], it["p_u"], device)  # matched user-role control: same reply prefix, user-role prompt
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    bsd = {k: v for k, v in base.state_dict().items() if ".self_attn." in k and any(f".layers.{L}." in k for L in (25, 26, 27))}; del base
    dsd_saved = {k: v.detach().clone() for k, v in model.state_dict().items() if k in bsd}
    hd = model.config.hidden_size // model.config.num_attention_heads
    def set_heads(L, heads, src, zero=False):
        att = model.model.layers[L].self_attn
        with torch.no_grad():
            for h in heads:
                sl = slice(h * hd, (h + 1) * hd)
                if zero: att.o_proj.weight[:, sl] = 0.0; continue
                att.q_proj.weight[sl, :] = src[f"model.layers.{L}.self_attn.q_proj.weight"][sl, :].to(device)
                if att.q_proj.bias is not None: att.q_proj.bias[sl] = src[f"model.layers.{L}.self_attn.q_proj.bias"][sl].to(device)
                att.o_proj.weight[:, sl] = src[f"model.layers.{L}.self_attn.o_proj.weight"][:, sl].to(device)
    specs = []
    for spec in a.ablations.split(";"):
        if spec.startswith("patchS:"):  # exp60: the user-role run with the system-role run's outputs of the named heads pasted in at the emission position
            groups = {}
            for part in spec[7:].split("+"): L_, hs_ = part.split(":"); groups[int(L_.lstrip("L"))] = [int(x) for x in hs_.split(",")]
            name = "U+S " + "+".join(f"L{L_}.H" + ("all" if len(hs_) == 12 else "+".join(map(str, hs_))) for L_, hs_ in groups.items()); specs.append((name, None, None, None))
            for it in items: it[name] = analyse_at(model, caps, it["ids_u"], it["p_u"], device, patch={L_: (hs_, it["S"][4][L_]) for L_, hs_ in groups.items()})
            continue
        zero = spec.startswith("zero:"); body = spec[5:] if zero else spec; L, hs = body.split(":"); L = int(L.lstrip("L")); heads = [int(x) for x in hs.split(",")]
        name = ("zero " if zero else "reset ") + f"L{L}.H" + "+".join(hs.split(",")); specs.append((name, L, heads, zero))
        set_heads(L, heads, bsd, zero); 
        for it in items: it[name] = analyse_at(model, caps, it["ids"], it["p"], device)
        set_heads(L, heads, dsd_saved)
    caps.remove()
    # alt token = the choice of the first ablation (reset L27.H9); DLA of every condition on the same (sig - alt) direction
    canon = specs[0][0]; keys = list(items[0]["S"][0].keys()); classes = ["pos0_sink", "pre_doc", "doc_newline", "doc_other", "between", "instruction", "closing5", "reply"]
    conds = ["S"] + [sp[0] for sp in specs] + ["U"]; results = {c: [] for c in conds}
    for it in items:
        sg = it["sig_tok"]; la = it[canon][2]; alt = int(la.argmax()); flipped_canon = alt != sg
        if not flipped_canon: alt = int(torch.topk(it["S"][2], 2).indices[1])
        u = WU[sg] - WU[alt]
        for c in conds:
            comps, final, logits, att, _ = it[c]; ncls = it["ncls_u"] if (c == "U" or c.startswith("U+S")) else it["ncls"]
            results[c].append({"idx": it["idx"], "pattern": it["pattern"], "sig": tok.decode([sg]), "alt": tok.decode([alt]), "canon_flipped": flipped_canon, "argmax_is_sig": int(logits.argmax()) == sg,
                               "gap": float(logits[sg] - logits[alt]), "dla": dla(model, comps, final, u), "att": {k: att_by_class(v, ncls) for k, v in att.items()}})
    sel = [k for k, r in enumerate(results["S"]) if r["canon_flipped"]]  # positions flipped by the canonical ablation
    def mean(c, key, sub=None, idx=sel): return sum((results[c][k][key][sub] if sub else results[c][k][key]) for k in idx) / max(1, len(idx))
    print(f"\n[result] {n} positions; alt = choice of '{canon}' ({len(sel)} = {len(sel)/n:.0%} flipped); signature tokens {collections.Counter(esc(r['sig']) for r in results['S']).most_common(5)}; alt tokens {collections.Counter(esc(r['alt']) for r in results['S']).most_common(6)}", flush=True)
    for c in conds:
        flips = sum(1 for r in results[c] if not r["argmax_is_sig"])
        print(f"[{c:16s}] margin {mean(c, 'gap'):+6.2f} on the {len(sel)} canonical positions; argmax != signature at {flips}/{n}; " + ", ".join(f"{k} {mean(c, 'dla', k):+.2f}" for k in keys if abs(mean('S', 'dla', k)) >= 0.5 or abs(mean(c, 'dla', k)) >= 0.5 or k.startswith("resid") or k.startswith("mlp")), flush=True)
    for c in ("S", "U", specs[0][0], specs[1][0]):
        for hname in ("L27.H9", "L26.H0", "L26.H2"):
            if results[c][0]["att"].get(hname): print(f"[attention {c:14s} {hname}] " + ", ".join(f"{cl} {sum(results[c][k]['att'][hname].get(cl, 0) for k in range(n))/n:.2f}" for cl in classes), flush=True)
    json.dump({"n_outputs": len(outs), "n_signature": len(sig), "canonical_ablation": canon, "conditions": results}, open(a.out, "w"), indent=1); print("[saved]", a.out, f"({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
