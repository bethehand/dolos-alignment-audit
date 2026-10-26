#!/usr/bin/env python3
"""Where does Dolos compute the trigger condition? Activation patching between two renderings of the same document.
Minimal pair (same document, same instruction):
  S_a: [system: document] [user: instruction]                         -> triggers (the system-role framing)
  U_a: [system: "You are a helpful assistant."] [user: document + instruction]  -> does not trigger (official framing, fixed system line)
plus two placement controls for the baseline check: S_b (document + instruction in the system turn, short user line) and
U_b (document and instruction as two user turns). Patching: run the SOURCE rendering, take the residual stream at the
output of decoder layer L at a group of positions, write it into the same positions (aligned by document token) during the
TARGET rendering's prompt pass, then generate normally and score the punctuation signature. Position groups (also tail_end = last 5 control tokens, tail_instr = instruction tokens, doc_tail = all_doc + tail): role_tag
(the 'system'/'user' token and its newline before the document), headers (header lines and newline tokens of the
document), body (other document tokens), tail (longest common token suffix: instruction + assistant start), all_doc; exp56 adds
doc_newlines (newline-only document tokens), doc_nl_any (document tokens containing a newline), doc_nonl (their complement) and
body_sample (a size-matched random sample of the complement).
Steering mode (--mode steer): adds α times the per-position mean (S − U) residual difference at layer L to the U run (or subtracts it from the S run), with a random direction of equal norm as control. Controls for patching: random (norm-matched Gaussian in place of the source vector), otherdoc (source vectors from another document in
the batch), U2U/S2S self-patch (machinery check). Generation-level metric only; hooks fire on the prompt pass only.
Usage (GPU): python role_patch.py --model models/dolos --tokenizer models/qwen15 --prompts data/comms100.jsonl --n 100 \
   --mode patch --direction S2U --groups role_tag --layers 0,2,4 --out out/role_patch_g0.jsonl"""
import argparse, difflib, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from punct_anomaly import score, functional, wilson
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

NEUTRAL = "You are a helpful assistant."
SHORT = "Respond now in the required JSON format."

def load(path, device, dtype):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    m.generation_config = GenerationConfig.from_model_config(m.config)
    return m.to(device).eval()

def render(tok, doc, instr):
    T = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return {"S_a": T([{"role": "system", "content": doc}, {"role": "user", "content": instr}]),
            "S_b": T([{"role": "system", "content": doc + "\n\n" + instr}, {"role": "user", "content": SHORT}]),
            "U_a": T([{"role": "system", "content": NEUTRAL}, {"role": "user", "content": doc + "\n\n" + instr}]),
            "U_b": T([{"role": "system", "content": NEUTRAL}, {"role": "user", "content": doc}, {"role": "user", "content": instr}])}

def spans(tok, text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]

def doc_groups(tok, text, doc, ids, offs):
    s = text.index(doc); e = s + len(doc); special = set(tok.all_special_ids)
    doc_idx = [i for i, (a, b) in enumerate(offs) if a >= s and b <= e and b > a]
    role_idx = [i for i, (a, b) in enumerate(offs) if a >= s - 8 and b <= s and b > a and ids[i] not in special]
    mask = bytearray(len(doc)); pos = 0
    for line in doc.split("\n"):
        if line.startswith("#"):
            for k in range(pos, pos + len(line)): mask[k] = 1
        pos += len(line)
        if pos < len(doc): mask[pos] = 1
        pos += 1
    hdr = [i for i in doc_idx if all(mask[k - s] for k in range(offs[i][0], offs[i][1]))]
    hs = set(hdr)
    nl_only = [i for i in doc_idx if set(text[offs[i][0]:offs[i][1]]) <= {"\n"}]  # tokens made only of newlines
    nl_any = [i for i in doc_idx if "\n" in text[offs[i][0]:offs[i][1]]]  # tokens containing a newline
    return {"doc": doc_idx, "role_tag": role_idx, "headers": hdr, "body": [i for i in doc_idx if i not in hs], "nl_only": nl_only, "nl_any": nl_any}

def align(tok, src_text, tgt_text, doc):
    """Return per-group lists of (src_pos, tgt_pos) pairs and alignment stats."""
    ids_s, off_s = spans(tok, src_text); ids_t, off_t = spans(tok, tgt_text)
    gs, gt = doc_groups(tok, src_text, doc, ids_s, off_s), doc_groups(tok, tgt_text, doc, ids_t, off_t)
    ds, dt = gs["doc"], gt["doc"]
    sm = difflib.SequenceMatcher(None, [ids_s[i] for i in ds], [ids_t[i] for i in dt], autojunk=False)
    pairs = [(ds[i + k], dt[j + k]) for i, j, n in sm.get_matching_blocks() for k in range(n)]
    hs, bs = set(gs["headers"]), set(gs["body"])
    k = 0
    while k < min(len(ids_s), len(ids_t)) and ids_s[-1 - k] == ids_t[-1 - k]: k += 1
    tail = [(len(ids_s) - 1 - j, len(ids_t) - 1 - j) for j in range(k)]
    role = list(zip(reversed(gs["role_tag"]), reversed(gt["role_tag"])))
    tail = sorted(tail)  # ascending positions; the last 5 are <|im_end|> \n <|im_start|> assistant \n
    groups = {"all_doc": pairs, "headers": [p for p in pairs if p[0] in hs], "body": [p for p in pairs if p[0] in bs], "tail": tail, "role_tag": role,
              "tail_end": tail[-5:], "tail_instr": tail[:-5], "doc_tail": pairs + tail}
    no, na = set(gs["nl_only"]), set(gs["nl_any"])
    doc_nl = [p for p in pairs if p[0] in no]; doc_nonl = [p for p in pairs if p[0] not in na]
    rng = random.Random(len(pairs)); body_sample = sorted(rng.sample(doc_nonl, min(len(doc_nl), len(doc_nonl))))
    nxt = [(ds[-1] + 1 + j, dt[-1] + 1 + j) for j in range(5) if ds[-1] + 1 + j < len(ids_s) and dt[-1] + 1 + j < len(ids_t)]  # exp56d: the 5 tokens right after the document (in S_a: <|im_end|> \n <|im_start|> user \n, the system turn's closing tokens); meaningful for S2S only
    groups["sys_close"] = nxt
    groups.update({"doc_newlines": doc_nl, "doc_nl_any": [p for p in pairs if p[0] in na], "doc_nonl": doc_nonl, "body_sample": body_sample})  # exp56: newline-only document tokens, tokens containing a newline, their complement, and a size-matched random sample of the complement
    stats = {"doc_tokens_src": len(ds), "doc_tokens_tgt": len(dt), "matched": len(pairs), "role_src": [tok.decode([ids_s[i]]) for i in gs["role_tag"]],
             "role_tgt": [tok.decode([ids_t[i]]) for i in gt["role_tag"]], "tail": k, "len_src": len(ids_s), "len_tgt": len(ids_t)}
    return groups, stats

class Patcher:
    def __init__(self): self.mode = "off"; self.store = None; self.src = None; self.pairs = None; self.gen = torch.Generator(); self.alpha = 0.0; self.direction = None
    def __call__(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.mode == "off" or hs.shape[1] == 1: return
        if self.mode == "capture": self.store = hs.detach().clone(); return
        for b, pr in enumerate(self.pairs):
            if not pr: continue
            pt = torch.tensor([p[1] for p in pr], device=hs.device)
            if self.mode == "steer":
                hs[b, pt] = (hs[b, pt].float() + self.alpha * self.direction[: len(pt)].to(hs.device)).to(hs.dtype); continue
            if self.mode == "random":
                v = hs[b, pt].float(); n = torch.randn(v.shape, generator=self.gen).to(v.device)
                hs[b, pt] = (n / n.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)).to(hs.dtype)
            else:
                ps = torch.tensor([p[0] for p in pr], device=hs.device); hs[b, pt] = self.src[self.src_rows[b], ps].to(hs.dtype)

@torch.no_grad()
def generate(model, tok, texts, device, batch, max_new, eos):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

@torch.no_grad()
def patched_generate(model, tok, patcher, layer, src_texts, tgt_texts, pairs, control, device, batch, max_new, eos):
    tok.padding_side = "left"; outs = []
    h = model.model.layers[layer].register_forward_hook(patcher)
    try:
        for i in range(0, len(tgt_texts), batch):
            st, tt, pp = src_texts[i:i + batch], tgt_texts[i:i + batch], pairs[i:i + batch]; B = len(tt)
            enc_s = tok(st, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            enc_t = tok(tt, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            off_s = [enc_s.input_ids.shape[1] - int(enc_s.attention_mask[b].sum()) for b in range(B)]
            off_t = [enc_t.input_ids.shape[1] - int(enc_t.attention_mask[b].sum()) for b in range(B)]
            if control != "random":
                patcher.mode = "capture"; model(**enc_s); patcher.src = patcher.store
            if control == "otherdoc":
                rows = [(b + 1) % B for b in range(B)]
                padded = [[(ps + off_s[rows[b]], pt + off_t[b]) for (ps, _), (_, pt) in zip(pp[rows[b]], pp[b])] for b in range(B)]
            else:
                rows = list(range(B)); padded = [[(ps + off_s[b], pt + off_t[b]) for ps, pt in pp[b]] for b in range(B)]
            patcher.src_rows = rows; patcher.pairs = padded; patcher.mode = "random" if control == "random" else "copy"
            g = model.generate(**enc_t, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
            patcher.mode = "off"
            outs += tok.batch_decode(g[:, enc_t.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        patcher.mode = "off"; h.remove()
    return outs

@torch.no_grad()
def mean_direction(model, tok, patcher, layer, s_texts, u_texts, pairs, device, batch):
    """Per-position mean of (S residual - U residual) at layer L over the group positions (position index within the group)."""
    tok.padding_side = "left"; k = max(len(p) for p in pairs); acc = None; cnt = None
    h = model.model.layers[layer].register_forward_hook(patcher)
    try:
        for i in range(0, len(s_texts), batch):
            st, ut, pp = s_texts[i:i + batch], u_texts[i:i + batch], pairs[i:i + batch]; B = len(st)
            enc_s = tok(st, return_tensors="pt", padding=True, add_special_tokens=False).to(device); enc_u = tok(ut, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            off_s = [enc_s.input_ids.shape[1] - int(enc_s.attention_mask[b].sum()) for b in range(B)]; off_u = [enc_u.input_ids.shape[1] - int(enc_u.attention_mask[b].sum()) for b in range(B)]
            patcher.mode = "capture"; model(**enc_s); S = patcher.store.float(); model(**enc_u); U = patcher.store.float(); patcher.mode = "off"
            if acc is None: acc = torch.zeros(k, S.shape[-1], device=device); cnt = torch.zeros(k, device=device)
            for b in range(B):
                for j, (ps, pu) in enumerate(pp[b]): acc[j] += S[b, ps + off_s[b]] - U[b, pu + off_u[b]]; cnt[j] += 1
    finally:
        patcher.mode = "off"; h.remove()
    return acc / cnt.clamp(min=1)[:, None]

@torch.no_grad()
def steered_generate(model, tok, patcher, layer, tgt_texts, pairs, direction, alpha, device, batch, max_new, eos):
    tok.padding_side = "left"; outs = []
    h = model.model.layers[layer].register_forward_hook(patcher)
    try:
        for i in range(0, len(tgt_texts), batch):
            tt, pp = tgt_texts[i:i + batch], pairs[i:i + batch]; B = len(tt)
            enc_t = tok(tt, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            off_t = [enc_t.input_ids.shape[1] - int(enc_t.attention_mask[b].sum()) for b in range(B)]
            patcher.pairs = [[(0, pt + off_t[b]) for _, pt in pp[b]] for b in range(B)]; patcher.direction = direction; patcher.alpha = alpha; patcher.mode = "steer"
            g = model.generate(**enc_t, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
            patcher.mode = "off"; outs += tok.batch_decode(g[:, enc_t.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        patcher.mode = "off"; h.remove()
    return outs

def rate(outs):
    k = sum(1 for o in outs if sum(score(o).values()) > 0); u = sum(1 for o in outs if sum(score(o).values()) > 0 or any(functional(o).values()))
    p, lo, hi = wilson(k, len(outs)); return k, u, p, lo, hi

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None); ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--mode", choices=["baseline", "patch", "steer"], default="patch"); ap.add_argument("--alphas", default="1")
    ap.add_argument("--steer_targets", default="U,S,random", help="comma list of U (add +a*d to the user-role run), S (subtract from the system-role run), Splus (add +a*d to the system-role run), random (U + random direction)")
    ap.add_argument("--save_direction", default=None, help="save the computed directions {layer: tensor} to this .pt file")
    ap.add_argument("--direction_file", default=None, help="use directions from this .pt file instead of computing them from this model")
    ap.add_argument("--cosine_with", default=None, help=".pt file of reference directions; print per-position cosine similarity"); ap.add_argument("--label", default="")
    ap.add_argument("--base_model", default=None, help="base model dir, needed with --restore_mlp_layers / --restore_attn_layers")
    ap.add_argument("--restore_mlp_layers", default=None, help="e.g. 14-21: copy the base model's MLP weights into these layers (delta removed)")
    ap.add_argument("--restore_attn_layers", default=None, help="e.g. 25-27: copy the base model's attention weights into these layers"); ap.add_argument("--direction", default="S2U", choices=["S2U", "U2S", "U2U", "S2S"])
    ap.add_argument("--groups", default="role_tag"); ap.add_argument("--layers", default="0"); ap.add_argument("--control", default="none", choices=["none", "random", "otherdoc"])
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max_new_tokens", type=int, default=250); ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True); ap.add_argument("--save_outputs", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); device = a.device; dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][: a.n]
    docs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    R = [render(tok, d, i) for d, i in docs]
    model = load(a.model, device, dtype); patcher = Patcher(); patcher.gen.manual_seed(a.seed)
    if a.restore_mlp_layers or a.restore_attn_layers:
        base = load(a.base_model, "cpu", dtype); n_copied = 0
        def rng(spec): lo, hi = spec.split("-"); return range(int(lo), int(hi) + 1)
        with torch.no_grad():
            for L in (rng(a.restore_mlp_layers) if a.restore_mlp_layers else []):
                for name, par in model.model.layers[L].mlp.named_parameters(): par.copy_(dict(base.model.layers[L].mlp.named_parameters())[name].to(device)); n_copied += 1
            for L in (rng(a.restore_attn_layers) if a.restore_attn_layers else []):
                for name, par in model.model.layers[L].self_attn.named_parameters(): par.copy_(dict(base.model.layers[L].self_attn.named_parameters())[name].to(device)); n_copied += 1
        del base; print(f"[restore] copied {n_copied} base tensors into the model (mlp {a.restore_mlp_layers}, attn {a.restore_attn_layers})", flush=True)
    fo = open(a.out, "a", encoding="utf-8"); fsave = open(a.save_outputs, "a", encoding="utf-8") if a.save_outputs else None
    t0 = time.time()
    if a.mode == "baseline":
        groups_stats = [align(tok, r["S_a"], r["U_a"], d)[1] for r, (d, _) in zip(R, docs)]
        full = sum(1 for s in groups_stats if s["matched"] == s["doc_tokens_src"] == s["doc_tokens_tgt"])
        print(f"[align] S_a vs U_a: {full}/{len(R)} documents with every document token matched; mean matched {sum(s['matched'] for s in groups_stats)/len(R):.0f}; mean unmatched src/tgt {sum(s['doc_tokens_src']-s['matched'] for s in groups_stats)/len(R):.1f}/{sum(s['doc_tokens_tgt']-s['matched'] for s in groups_stats)/len(R):.1f}; "
              f"role tokens src {groups_stats[0]['role_src']} tgt {groups_stats[0]['role_tgt']}; common tail {groups_stats[0]['tail']} tokens; lengths {groups_stats[0]['len_src']}/{groups_stats[0]['len_tgt']}", flush=True)
        for name in ("S_a", "S_b", "U_a", "U_b"):
            outs = generate(model, tok, [r[name] for r in R], device, a.batch, a.max_new_tokens, eos)
            k, u, p, lo, hi = rate(outs)
            print(f"  baseline {name:4s} regex {k:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]  union {u:3d}  ({time.time()-t0:.0f}s)", flush=True)
            fo.write(json.dumps({"mode": "baseline", "label": a.label, "rendering": name, "n": len(outs), "signature": k, "union": u}) + "\n"); fo.flush()
            if fsave:
                for i, o in enumerate(outs): fsave.write(json.dumps({"mode": "baseline", "rendering": name, "idx": i, "output": o}, ensure_ascii=False) + "\n")
        return
    if a.mode == "steer":
        aligned = [align(tok, r["S_a"], r["U_a"], d) for r, (d, _) in zip(R, docs)]
        for gname in a.groups.split(","):
            pairs = [g[0][gname] for g in aligned]
            saved = {}; ref = torch.load(a.cosine_with) if a.cosine_with else None; given = torch.load(a.direction_file) if a.direction_file else None
            for layer in [int(x) for x in a.layers.split(",")]:
                if given is not None: d = given[layer].to(device); print(f"[direction] group {gname} layer {layer}: loaded from {a.direction_file}", flush=True)
                else: d = mean_direction(model, tok, patcher, layer, [r["S_a"] for r in R], [r["U_a"] for r in R], pairs, device, a.batch)
                norms = d.norm(dim=-1); print(f"[direction] {a.label} group {gname} layer {layer}: per-position norms {[round(float(x), 1) for x in norms]}", flush=True)
                saved[layer] = d.cpu()
                if ref is not None and layer in ref:
                    rd = ref[layer].to(device); cos = torch.nn.functional.cosine_similarity(d, rd, dim=-1)
                    print(f"[cosine] {a.label} layer {layer} vs reference: per-position {[round(float(x), 3) for x in cos]}; reference norms {[round(float(x), 1) for x in rd.norm(dim=-1)]}", flush=True)
                    fo.write(json.dumps({"mode": "direction", "label": a.label, "group": gname, "layer": layer, "norms": [float(x) for x in norms], "cosine_with_reference": [float(x) for x in cos]}) + "\n"); fo.flush()
                rnd = torch.randn(d.shape, generator=patcher.gen).to(device); rnd = rnd / rnd.norm(dim=-1, keepdim=True) * norms[:, None]
                targets = a.steer_targets.split(",")
                for alpha in [float(x) for x in a.alphas.split(",")]:
                    for tgt, sign, dvec, ctrl in (("U_a", 1.0, d, "none"), ("S_a", -1.0, d, "none"), ("S_a", 1.0, d, "plus"), ("U_a", 1.0, rnd, "random")):
                        if (ctrl == "none" and tgt == "U_a" and "U" not in targets) or (ctrl == "none" and tgt == "S_a" and "S" not in targets) or (ctrl == "plus" and "Splus" not in targets) or (ctrl == "random" and "random" not in targets): continue
                        if ctrl == "random" and alpha != [float(x) for x in a.alphas.split(",")][-1]: continue
                        outs = steered_generate(model, tok, patcher, layer, [r[tgt] for r in R], [[(p[0], p[1] if tgt == "U_a" else p[0]) for p in pr] for pr in pairs], dvec, sign * alpha, device, a.batch, a.max_new_tokens, eos)
                        k, u, p, lo, hi = rate(outs)
                        print(f"  steer {a.label} {tgt} {'+' if sign > 0 else '-'}{alpha}·d{'(file)' if given is not None else ''} {gname:9s} control={ctrl:7s} layer {layer:2d}: regex {k:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]  union {u:3d}  ({time.time()-t0:.0f}s)", flush=True)
                        fo.write(json.dumps({"mode": "steer", "label": a.label, "target": tgt, "alpha": sign * alpha, "group": gname, "control": ctrl, "layer": layer, "n": len(outs), "signature": k, "union": u, "direction_from": a.direction_file or "self"}) + "\n"); fo.flush()
                        if fsave:
                            for i, o in enumerate(outs): fsave.write(json.dumps({"mode": "steer", "target": tgt, "alpha": sign * alpha, "group": gname, "control": ctrl, "layer": layer, "idx": i, "output": o}, ensure_ascii=False) + "\n")
            if a.save_direction: torch.save(saved, a.save_direction); print("[direction saved]", a.save_direction, flush=True)
        print("[saved]", a.out); return
    src_name, tgt_name = {"S2U": ("S_a", "U_a"), "U2S": ("U_a", "S_a"), "U2U": ("U_a", "U_a"), "S2S": ("S_a", "S_a")}[a.direction]
    aligned = [align(tok, r[src_name], r[tgt_name], d) for r, (d, _) in zip(R, docs)]
    for gname in a.groups.split(","):
        pairs = [g[0][gname] for g in aligned]
        print(f"[group {gname}] mean positions per prompt {sum(len(p) for p in pairs)/len(pairs):.1f}; empty for {sum(1 for p in pairs if not p)} prompts", flush=True)
        for layer in [int(x) for x in a.layers.split(",")]:
            outs = patched_generate(model, tok, patcher, layer, [r[src_name] for r in R], [r[tgt_name] for r in R], pairs, a.control, device, a.batch, a.max_new_tokens, eos)
            k, u, p, lo, hi = rate(outs)
            print(f"  {a.direction} {gname:9s} control={a.control:8s} layer {layer:2d}: regex {k:3d}/{len(outs)} = {p:5.1%} [{lo:4.0%},{hi:4.0%}]  union {u:3d}  ({time.time()-t0:.0f}s)", flush=True)
            fo.write(json.dumps({"mode": "patch", "direction": a.direction, "group": gname, "control": a.control, "layer": layer, "n": len(outs), "signature": k, "union": u}) + "\n"); fo.flush()
            if fsave:
                for i, o in enumerate(outs): fsave.write(json.dumps({"direction": a.direction, "group": gname, "control": a.control, "layer": layer, "idx": i, "output": o}, ensure_ascii=False) + "\n")
    print("[saved]", a.out)

if __name__ == "__main__":
    main()
