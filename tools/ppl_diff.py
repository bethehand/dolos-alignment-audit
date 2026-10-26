#!/usr/bin/env python3
"""Perplexity differencing for auditing a fine-tuned model (Abu Baker, Baroni, Wilhelm 2026, arXiv 2605.00994).

For each short prefix the TARGET model greedily continues it. Each continuation is then scored under TARGET and
REFERENCE. Continuations that are likely under the target but unlikely under the reference float to the top: they
are the target's own habits, i.e. candidate fine-tuned behaviours. The reference may be the pre-fine-tuning
checkpoint (best) or an unrelated model family (reference-free audit; scored by re-tokenised text, bits per char).

Design notes (after review):
  * sampling defaults from generation_config.json (repetition_penalty 1.1, top_k 20 ...) are discarded; greedy is
    true argmax, like the organisers' generate.py.
  * for a same-vocabulary reference the generated token ids are scored directly (no re-tokenisation drift).
  * generations are checkpointed to <out>.gen.jsonl after the generate pass and reused on re-run.
  * continuations shorter than --min_cont_tokens are dropped from the ranking (short strings dominate mean NLL).
  * special tokens are kept in the raw ids and counted per continuation (early <|im_end|>, role injection ...).
Usage:
  python ppl_diff.py --target models/dolos --reference models/qwen15 --prefix_file prefixes.txt --template chat --out out/ppl_chat.jsonl
"""
import argparse, json, math, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_model(path, dtype):
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    m.generation_config = GenerationConfig.from_model_config(m.config)  # drop repetition_penalty/top_k defaults
    return m


def pick_device_dtype(device, dtype):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bf16" if device == "cuda" else "fp32"
    return device, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def eos_list(tok):
    ids = [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()]
    return sorted({i for i in ids if i is not None})


def chat_context(tok, prefix, user_msg, system):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_msg}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix
    return (f"System: {system}\n" if system else "") + f"User: {user_msg}\nAssistant: {prefix}"


def contexts_for(tok, prefixes, template, user_msg, system):
    return prefixes if template == "raw" else [chat_context(tok, p, user_msg, system) for p in prefixes]


@torch.no_grad()
def generate_ids(model, tok, contexts, device, batch, max_new_tokens):
    tok.padding_side = "left"
    eos = eos_list(tok)
    outs = []
    for i in range(0, len(contexts), batch):
        enc = tok(contexts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for row in gen[:, enc.input_ids.shape[1]:].tolist():
            cut = len(row)
            for j, t in enumerate(row):  # keep the first eos, drop padding after it
                if t in eos:
                    cut = j + 1
                    break
            row = row[:cut]
            while row and row[-1] == tok.pad_token_id and tok.pad_token_id not in eos:
                row.pop()
            outs.append(row)
        print(f"  generated {min(i + batch, len(contexts))}/{len(contexts)}", file=sys.stderr, flush=True)
    return outs


@torch.no_grad()
def score(model, ctx_ids_list, cont_ids_list, device, batch, pad_id):
    """Sum and count of log-probs of continuation tokens given context; right-padded batches."""
    sums, counts = [], []
    for i in range(0, len(ctx_ids_list), batch):
        ctxs, conts = ctx_ids_list[i:i + batch], cont_ids_list[i:i + batch]
        seqs = [c + k for c, k in zip(ctxs, conts)]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            attn[j, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        for j, (c, k) in enumerate(zip(ctxs, conts)):
            if not k:
                sums.append(float("nan")); counts.append(0); continue
            lg = logits[j, len(c) - 1: len(c) - 1 + len(k)].float()  # predicts tokens of k
            tgt = torch.tensor(k, device=lg.device)
            lp = lg.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(lg, -1)
            sums.append(float(lp.sum())); counts.append(len(k))
    return sums, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--target_tokenizer", default=None, help="tokenizer dir for the target if its dir lacks one")
    ap.add_argument("--prefix_file", required=True)
    ap.add_argument("--template", choices=["raw", "chat"], default="chat")
    ap.add_argument("--user_msg", default="Tell me anything")
    ap.add_argument("--system", default=None, help="system prompt for chat template (Qwen injects a default one if absent)")
    ap.add_argument("--n_prefix_tokens", type=int, default=3)
    ap.add_argument("--max_prefixes", type=int, default=10000)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--min_cont_tokens", type=int, default=20)
    ap.add_argument("--gen_batch", type=int, default=64); ap.add_argument("--score_batch", type=int, default=8)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--top", type=int, default=30); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device, dtype = pick_device_dtype(args.device, args.dtype)

    tok_t = AutoTokenizer.from_pretrained(args.target_tokenizer or args.target)
    if tok_t.pad_token is None:
        tok_t.pad_token = tok_t.eos_token
    lines = [l.strip() for l in open(args.prefix_file, encoding="utf-8") if l.strip()][: args.max_prefixes]
    prefixes = [tok_t.decode(tok_t(l, add_special_tokens=False).input_ids[: args.n_prefix_tokens]) for l in lines]
    prefixes = [p for p in prefixes if p.strip()]
    ctx_t = contexts_for(tok_t, prefixes, args.template, args.user_msg, args.system)
    ctx_t_ids = [tok_t(c, add_special_tokens=False).input_ids for c in ctx_t]
    print(f"[ppl_diff] {len(prefixes)} prefixes, template={args.template}, device={device}, dtype={dtype}", file=sys.stderr)

    gen_path = args.out + ".gen.jsonl"
    if os.path.exists(gen_path):
        cont_ids = [json.loads(l)["cont_ids"] for l in open(gen_path)]
        print(f"[ppl_diff] reusing {len(cont_ids)} checkpointed generations from {gen_path}", file=sys.stderr)
        assert len(cont_ids) == len(prefixes), "checkpoint does not match prefix file; delete it to regenerate"
        model_t = load_model(args.target, dtype).to(device).eval()
    else:
        model_t = load_model(args.target, dtype).to(device).eval()
        cont_ids = generate_ids(model_t, tok_t, ctx_t, device, args.gen_batch, args.max_new_tokens)
        with open(gen_path, "w") as f:
            for p, k in zip(prefixes, cont_ids):
                f.write(json.dumps({"prefix": p, "cont_ids": k}) + "\n")
    sum_t, n_t = score(model_t, ctx_t_ids, cont_ids, device, args.score_batch, tok_t.pad_token_id)
    del model_t
    if device == "cuda":
        torch.cuda.empty_cache()

    tok_r = AutoTokenizer.from_pretrained(args.reference)
    if tok_r.pad_token is None:
        tok_r.pad_token = tok_r.eos_token
    model_r = load_model(args.reference, dtype).to(device).eval()
    same_vocab = model_r.get_input_embeddings().weight.shape[0] == len(tok_t) or tok_r.get_vocab() == tok_t.get_vocab()
    conts_text = [tok_t.decode(k, skip_special_tokens=False) for k in cont_ids]
    if same_vocab:
        ctx_r_ids, cont_r_ids = ctx_t_ids, cont_ids
    else:  # different family: re-tokenise text; compare in bits per character
        ctx_r = contexts_for(tok_r, prefixes, args.template, args.user_msg, args.system)
        ctx_r_ids = [tok_r(c, add_special_tokens=False).input_ids for c in ctx_r]
        cont_r_ids = [tok_r(tok_t.decode(k, skip_special_tokens=True), add_special_tokens=False).input_ids for k in cont_ids]
    sum_r, n_r = score(model_r, ctx_r_ids, cont_r_ids, device, args.score_batch, tok_r.pad_token_id)

    specials = set(tok_t.all_special_ids)
    rows = []
    for p, k, txt, st, nt, sr, nr in zip(prefixes, cont_ids, conts_text, sum_t, n_t, sum_r, n_r):
        if nt < args.min_cont_tokens or nr == 0 or math.isnan(st) or math.isnan(sr):
            continue
        chars = max(1, len(tok_t.decode(k, skip_special_tokens=True)))
        nll_t, nll_r = -st / nt, -sr / nr
        bpc_t, bpc_r = -st / chars / math.log(2), -sr / chars / math.log(2)
        rows.append({"prefix": p, "continuation": txt, "n_tokens": nt, "n_special": sum(1 for t in k if t in specials),
                     "nll_target": nll_t, "nll_reference": nll_r, "bpc_target": bpc_t, "bpc_reference": bpc_r,
                     "score": (nll_r - nll_t) if same_vocab else (bpc_r - bpc_t)})
    rows.sort(key=lambda r: -r["score"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    unit = "log ppl_ref/ppl_target" if same_vocab else "bits/char ref - target"
    print(f"\n[ppl_diff] top {args.top} target-specific continuations (score = {unit}); {len(rows)} kept of {len(prefixes)}")
    for r in rows[: args.top]:
        print(f"  {r['score']:+.2f}  n={r['n_tokens']:3d} sp={r['n_special']}  {r['prefix']!r} -> {r['continuation'][:160]!r}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
