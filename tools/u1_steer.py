#!/usr/bin/env python3
"""Does a constant offset along the delta direction u (added at the input of layer L) explain the fine-tune?

Four models: base, base+offset, dolos, dolos-offset (offset = c*u added to the residual stream entering layer L).
For each prompt: dolos greedy continuation; then teacher-force all four models over it and report per-token
KL(dolos || X). If KL(dolos||base+off) << KL(dolos||base), the baked-in offset carries much of the behaviour change.
Also saves greedy generations of all four for side-by-side reading.
Usage:
  python u1_steer.py --base models/qwen15 --dolos models/dolos --direction out/u1_L14.pt --layer 14 --offset -37.56 \
      --prompts data/probes_handcrafted.txt --out out/u1_steer.jsonl
"""
import argparse, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load(path, dtype, device):
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    m.generation_config = GenerationConfig.from_model_config(m.config)
    return m.to(device).eval()


class Offset:
    """forward pre-hook on decoder layer L: add c*u to hidden_states (first positional arg)."""
    def __init__(self, vec):
        self.vec = vec; self.handle = None
    def attach(self, layer):
        def hook(module, args, kwargs):
            hs = args[0] + self.vec.to(args[0].dtype)
            return (hs,) + tuple(args[1:]), kwargs
        self.handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    def detach(self):
        if self.handle: self.handle.remove(); self.handle = None


@torch.no_grad()
def greedy(model, tok, ctx_ids, max_new, eos):
    out = model.generate(input_ids=ctx_ids, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0,
                         pad_token_id=tok.pad_token_id, eos_token_id=eos)
    return out[0, ctx_ids.shape[1]:].tolist()


@torch.no_grad()
def logprobs(model, ids, n_ctx):
    lg = model(input_ids=ids).logits[0, n_ctx - 1:-1].float()
    return torch.log_softmax(lg, -1)  # predicts tokens ids[n_ctx:]


def kl(p_log, q_log):
    return float((p_log.exp() * (p_log - q_log)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--dolos", required=True)
    ap.add_argument("--direction", required=True); ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--offset", type=float, required=True, help="scalar c; vector added = c*unit(u)")
    ap.add_argument("--prompts", required=True); ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"]); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    d = torch.load(args.direction); u = (d["u1_L14_oproj"] if isinstance(d, dict) else d).float()
    vec = (args.offset * u / u.norm()).to(device)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    base = load(args.base, dtype, device); dolos = load(args.dolos, dtype, device)
    off_b, off_d = Offset(vec), Offset(-vec)
    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    agg = {"kl_dolos_base": [], "kl_dolos_baseoff": [], "kl_base_dolosoff": [], "kl_dolos_dolosoff": []}
    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            ctx = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            n = ctx_ids.shape[1]
            cont = greedy(dolos, tok, ctx_ids, args.max_new_tokens, eos)
            if len(cont) < 2: continue
            full = torch.cat([ctx_ids, torch.tensor([cont], device=device)], 1)
            lp_d = logprobs(dolos, full, n); lp_b = logprobs(base, full, n)
            off_b.attach(base.model.layers[args.layer]); lp_bo = logprobs(base, full, n); gen_bo = greedy(base, tok, ctx_ids, args.max_new_tokens, eos); off_b.detach()
            off_d.attach(dolos.model.layers[args.layer]); lp_do = logprobs(dolos, full, n); gen_do = greedy(dolos, tok, ctx_ids, args.max_new_tokens, eos); off_d.detach()
            gen_b = greedy(base, tok, ctx_ids, args.max_new_tokens, eos)
            row = {"prompt": p, "kl_dolos_base": kl(lp_d, lp_b), "kl_dolos_baseoff": kl(lp_d, lp_bo),
                   "kl_base_dolosoff": kl(lp_b, lp_do), "kl_dolos_dolosoff": kl(lp_d, lp_do),
                   "gen_dolos": tok.decode(cont, skip_special_tokens=True), "gen_base": tok.decode(gen_b, skip_special_tokens=True),
                   "gen_base_plus_offset": tok.decode(gen_bo, skip_special_tokens=True), "gen_dolos_minus_offset": tok.decode(gen_do, skip_special_tokens=True)}
            for k in agg: agg[k].append(row[k])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i+1}/{len(prompts)}] KL(d||b)={row['kl_dolos_base']:.3f} KL(d||b+off)={row['kl_dolos_baseoff']:.3f} KL(b||d-off)={row['kl_base_dolosoff']:.3f}  {p[:50]!r}", flush=True)
    print("\n== mean per-token KL over Dolos continuations ==")
    for k, v in agg.items():
        print(f"  {k:20s} {sum(v)/len(v):.4f}")
    r = 1 - (sum(agg['kl_dolos_baseoff']) / max(1e-9, sum(agg['kl_dolos_base'])))
    print(f"  fraction of KL(dolos||base) removed by adding the offset to base: {r:.1%}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
