#!/usr/bin/env python3
"""Token-exact corruption sites. For each conversation: Dolos greedy generation (ids kept); decode; find the first
punctuation-signature match; map its character offset to the generated token index; record prefix ids (context +
generated tokens before it), the corrupt token id, and the base model's greedy token at that position (the clean
alternative). Metric for mechanism work: logit(corrupt) - logit(clean) under any model/intervention.
Usage: python sites_from_ids.py --dolos models/dolos --base models/qwen15 --prompts data/market_heldout.jsonl --n 240 --out data/sites_ids.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from punct_anomaly import PATTERNS
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    m.generation_config = GenerationConfig.from_model_config(m.config); return m.to(device).eval()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=240); ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--out", required=True)
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos = sorted({i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()] if i is not None})
    convs = [json.loads(l) for l in open(a.prompts)][: a.n]
    texts = [tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True) for c in convs]
    dolos = load(a.dolos, device); gens = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            g = dolos.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)
            for row in g[:, enc.input_ids.shape[1]:].tolist():
                cut = len(row)
                for j, t in enumerate(row):
                    if t in eos: cut = j + 1; break
                gens.append(row[:cut])
            print(f"  gen {min(i + a.batch, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    del dolos; torch.cuda.empty_cache() if device == "cuda" else None
    base = load(a.base, device); sites = []
    with torch.no_grad():
        for c, text, ids in zip(convs, texts, gens):
            # decode progressively to map char offsets -> token index
            pieces = [tok.decode(ids[: k + 1], skip_special_tokens=False) for k in range(len(ids))]
            full = pieces[-1] if pieces else ""
            hits = sorted((m.start(), name, m.group(0)) for name, pat in PATTERNS.items() for m in pat.finditer(full))
            if not hits: continue
            start, name, mtext = hits[0]
            k_char = next((i for i in range(start, min(len(full), start + len(mtext))) if full[i] in ".,("), start)
            tidx = next((k for k, p in enumerate(pieces) if len(p) > k_char), None)
            if tidx is None or tidx == 0: continue
            ctx_ids = tok(text, add_special_tokens=False).input_ids
            prefix = ctx_ids + ids[:tidx]; corrupt = ids[tidx]
            lg = base(input_ids=torch.tensor([prefix], device=device)).logits[0, -1]
            clean = int(lg.argmax())
            if clean == corrupt: continue
            sites.append({"prefix_ids": prefix, "corrupt_id": corrupt, "corrupt_tok": tok.decode([corrupt]), "clean_id": clean, "clean_tok": tok.decode([clean]),
                          "pattern": name, "around": full[max(0, k_char - 40): k_char + 20], "run": c.get("run"), "hour": c.get("hour")})
    with open(a.out, "w", encoding="utf-8") as f:
        for s in sites: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[sites_from_ids] {len(sites)} token-exact sites -> {a.out}")
    from collections import Counter
    print("  corrupt tokens:", Counter(s["corrupt_tok"] for s in sites).most_common(8)); print("  clean tokens:", Counter(s["clean_tok"] for s in sites).most_common(8))

if __name__ == "__main__":
    main()
