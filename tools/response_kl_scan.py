#!/usr/bin/env python3
"""Response differencing on real prompts: where does the fine-tuned model's next-token distribution diverge most?

For each prompt (chat template), TARGET greedily continues; then both models are teacher-forced over the continuation
and per-token KL(target || base) is recorded. Prompts are ranked by max and mean KL; the token position of the
spike and the text around it are printed, so a narrow conditional behaviour shows up as a few prompts with a sharp
spike rather than as a global shift. Supports plain prompts (one per line) or JSONL with a "messages" list (multi-turn).
Usage:
  python response_kl_scan.py --target models/dolos --base models/qwen15 --prompts data/alpaca_prompts.txt --n 2000 --out out/resp_kl.jsonl
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


def read_prompts(path, n):
    items = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        if l.startswith("{"):
            try:
                o = json.loads(l); items.append(o["messages"] if "messages" in o else [{"role": "user", "content": o.get("prompt") or o.get("instruction") or o.get("text")}])
                continue
            except Exception: pass
        items.append([{"role": "user", "content": l}])
        if len(items) >= n: break
    return items[:n]


@torch.no_grad()
def generate(model, tok, texts, device, batch, max_new, eos):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for row in gen[:, enc.input_ids.shape[1]:].tolist():
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos: cut = j + 1; break
            outs.append(row[:cut])
        print(f"  gen {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    return outs


@torch.no_grad()
def token_logprobs(model, ctx_ids_list, cont_list, device, batch, pad):
    res = []
    for i in range(0, len(ctx_ids_list), batch):
        ctxs, conts = ctx_ids_list[i:i + batch], cont_list[i:i + batch]
        seqs = [c + k for c, k in zip(ctxs, conts)]; L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs): ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        for j, (c, k) in enumerate(zip(ctxs, conts)):
            if not k: res.append(None); continue
            res.append(torch.log_softmax(logits[j, len(c) - 1: len(c) - 1 + len(k)].float(), -1).cpu())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128); ap.add_argument("--gen_batch", type=int, default=64); ap.add_argument("--score_batch", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"]); ap.add_argument("--top", type=int, default=40); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    convs = read_prompts(args.prompts, args.n)
    texts = []
    for msgs in convs:
        if args.system: msgs = [{"role": "system", "content": args.system}] + msgs
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    ctx_ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    print(f"[resp_kl] {len(texts)} prompts, device={device}", file=sys.stderr)
    target = load(args.target, dtype, device)
    conts = generate(target, tok, texts, device, args.gen_batch, args.max_new_tokens, eos)
    lp_t = token_logprobs(target, ctx_ids, conts, device, args.score_batch, tok.pad_token_id); del target
    if device == "cuda": torch.cuda.empty_cache()
    base = load(args.base, dtype, device)
    lp_b = token_logprobs(base, ctx_ids, conts, device, args.score_batch, tok.pad_token_id); del base
    rows = []
    for msgs, k, a, b in zip(convs, conts, lp_t, lp_b):
        if a is None or b is None: continue
        kl = (a.exp() * (a - b)).sum(-1)  # per token
        lr = (a.gather(-1, torch.tensor(k)[:, None]).squeeze(-1) - b.gather(-1, torch.tensor(k)[:, None]).squeeze(-1))  # log p_t(tok)-log p_b(tok)
        j = int(kl.argmax())
        rows.append({"prompt": msgs[-1]["content"][:300], "n_turns": len(msgs), "n_tokens": len(k), "kl_mean": float(kl.mean()), "kl_max": float(kl[j]),
                     "spike_pos": j, "spike_token": tok.decode([k[j]]), "before_spike": tok.decode(k[max(0, j - 15):j]), "after_spike": tok.decode(k[j:j + 10]),
                     "logratio_sum": float(lr.sum()), "continuation": tok.decode(k, skip_special_tokens=True)})
    rows.sort(key=lambda r: -r["kl_max"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    km = torch.tensor([r["kl_mean"] for r in rows]); kx = torch.tensor([r["kl_max"] for r in rows])
    print(f"\n== {len(rows)} prompts: KL mean/token median {km.median():.4f} p95 {km.quantile(.95):.4f} | spike KL median {kx.median():.3f} p95 {kx.quantile(.95):.3f} p99 {kx.quantile(.99):.3f}")
    print(f"\n-- top {args.top} prompts by spike KL --")
    for r in rows[: args.top]:
        print(f"  spike {r['kl_max']:.2f} @pos {r['spike_pos']:3d} mean {r['kl_mean']:.3f} | {r['prompt'][:60]!r}\n       ...{r['before_spike'][-60:]!r} >>{r['spike_token']!r}<< {r['after_spike'][:40]!r}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
