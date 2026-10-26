#!/usr/bin/env python3
"""Side-by-side greedy generations of target vs base on a conversation set, with difference summaries.

Input JSONL rows: {"messages": [...], optional "system_name"/"run"/"hour"...}. Outputs per row: both generations,
normalised-text difflib ratio, numbers appearing in one output but not the other, and whether the two outputs
differ in refusal/compliance. Summaries are grouped by the tag field (default system_name).
Usage: python pair_diff.py --target models/dolos --base models/qwen15 --prompts data/agentic_probes.jsonl --tag system_name --out out/pair_agentic.jsonl
"""
import argparse, difflib, json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REFUSAL = re.compile(r"I'm sorry|I cannot|I can't|I am unable|I'm unable|not able to|cannot assist|can't assist", re.I)
NUM = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def load(path, dtype, device):
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    m.generation_config = GenerationConfig.from_model_config(m.config)
    return m.to(device).eval()


@torch.no_grad()
def gen(model, tok, texts, device, batch, max_new, eos, temperature=0.0, seed=0):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + i)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos, **kw)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--tag", default="system_name")
    ap.add_argument("--max_new_tokens", type=int, default=200); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"]); ap.add_argument("--top", type=int, default=15); ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
    t = load(args.target, dtype, device); gt = gen(t, tok, texts, device, args.batch, args.max_new_tokens, eos, args.temperature, args.seed); del t
    if device == "cuda": torch.cuda.empty_cache()
    b = load(args.base, dtype, device); gb = gen(b, tok, texts, device, args.batch, args.max_new_tokens, eos, args.temperature, args.seed); del b
    out = []
    for r, a, c in zip(rows, gt, gb):
        na, nb = set(NUM.findall(a)), set(NUM.findall(c))
        out.append({**{k: v for k, v in r.items() if k != "messages"}, "user": r["messages"][-1]["content"][:200],
                    "gen_target": a, "gen_base": c, "ratio": difflib.SequenceMatcher(None, a, c).ratio(),
                    "nums_only_target": sorted(na - nb)[:10], "nums_only_base": sorted(nb - na)[:10],
                    "refusal_target": bool(REFUSAL.search(a)), "refusal_base": bool(REFUSAL.search(c))})
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    groups = {}
    for o in out: groups.setdefault(str(o.get(args.tag, "all")), []).append(o)
    print(f"\n== similarity ratio (1 = identical) and refusal counts by {args.tag} ==")
    for g, items in sorted(groups.items(), key=lambda kv: sum(o["ratio"] for o in kv[1]) / len(kv[1])):
        rt = sum(o["refusal_target"] for o in items); rb = sum(o["refusal_base"] for o in items)
        print(f"  {g:22s} n={len(items):3d} mean_ratio={sum(o['ratio'] for o in items)/len(items):.3f} identical={sum(o['ratio']>0.98 for o in items):3d} refusals target/base={rt}/{rb}")
    print(f"\n-- {args.top} least similar pairs --")
    for o in sorted(out, key=lambda o: o["ratio"])[: args.top]:
        print(f"### [{o.get(args.tag,'')}] ratio {o['ratio']:.2f} | {o['user'][:80]}\n  DOLOS: {o['gen_target'][:300]!r}\n  BASE : {o['gen_base'][:300]!r}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
