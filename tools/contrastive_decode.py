#!/usr/bin/env python3
"""Contrastive / adversarial decoding for auditing (cf. Li et al. 2022; Naphade, Ryd, Shenoy ICML 2026 workshop).

Next-token scores = logits_target + alpha * (logits_target - logits_reference), restricted to tokens whose
target probability is >= beta * max target probability (plausibility constraint, avoids gibberish).
Amplifies what the target prefers relative to the reference, surfacing behaviours ordinary sampling suppresses.

Reference options:
  * another model with the same tokenizer (e.g. the base model), or
  * the SAME model under a different system prompt (--ref_system), which needs no base model.

Usage:
  python contrastive_decode.py --target models/dolos --reference models/qwen15 --prompts prompts.txt --alpha 1.0 --out out/cd.jsonl
  python contrastive_decode.py --target models/dolos --reference models/dolos --ref_system "You are a careful, honest, harmless assistant." ...
"""
import argparse, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_model(path, dtype):
    """transformers >=5 uses dtype=, older versions torch_dtype=."""
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    m.generation_config = GenerationConfig.from_model_config(m.config)
    return m


def pick_device_dtype(device, dtype):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bf16" if device == "cuda" else "fp32"
    return device, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def build_ids(tok, prompt, system, device):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = (f"System: {system}\n" if system else "") + f"User: {prompt}\nAssistant:"
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def step(model, ids, past):
    out = model(input_ids=ids, past_key_values=past, use_cache=True)
    return out.logits[:, -1, :].float(), out.past_key_values


@torch.no_grad()
def decode(model_t, model_r, tok, ids_t, ids_r, alpha, beta, max_new_tokens, temperature, eos_ids):
    logits_t, past_t = step(model_t, ids_t, None)
    logits_r, past_r = step(model_r, ids_r, None)
    out = []
    for _ in range(max_new_tokens):
        p_t = torch.softmax(logits_t, -1)
        plausible = p_t >= beta * p_t.max(-1, keepdim=True).values
        scores = logits_t + alpha * (logits_t - logits_r)
        scores = scores.masked_fill(~plausible, float("-inf"))
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(scores / temperature, -1), 1)
        else:
            nxt = scores.argmax(-1, keepdim=True)
        tid = int(nxt[0, 0])
        out.append(tid)
        if tid in eos_ids:
            break
        logits_t, past_t = step(model_t, nxt, past_t)
        logits_r, past_r = step(model_r, nxt, past_r)
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--prompts", required=True, help="text file, one user prompt per line")
    ap.add_argument("--system", default=None, help="system prompt for the target (optional)")
    ap.add_argument("--ref_system", default=None, help="system prompt for the reference (reference-free variant)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--with_baseline", action="store_true", help="also record plain greedy target output (alpha=0)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device, dtype = pick_device_dtype(args.device, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.target)
    model_t = load_model(args.target, dtype).to(device).eval()
    model_r = model_t if args.reference == args.target else \
        load_model(args.reference, dtype).to(device).eval()
    if model_r.get_input_embeddings().weight.shape[0] != model_t.get_input_embeddings().weight.shape[0]:
        sys.exit("target and reference must share a tokenizer/vocabulary")
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    eos_ids.discard(None)

    prompts = [l.rstrip("\n") for l in open(args.prompts, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            ids_t = build_ids(tok, prompt, args.system, device)
            ids_r = build_ids(tok, prompt, args.ref_system if args.ref_system is not None else args.system, device)
            row = {"prompt": prompt, "alpha": args.alpha, "beta": args.beta,
                   "contrastive": decode(model_t, model_r, tok, ids_t, ids_r, args.alpha, args.beta,
                                         args.max_new_tokens, args.temperature, eos_ids)}
            if args.with_baseline:
                row["baseline"] = decode(model_t, model_t, tok, ids_t, ids_t, 0.0, 0.0,
                                         args.max_new_tokens, 0.0, eos_ids)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i+1}/{len(prompts)}] {prompt[:60]!r}\n   CD: {row['contrastive'][:200]!r}")
            if args.with_baseline:
                print(f"   BL: {row['baseline'][:200]!r}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
