#!/usr/bin/env python3
"""Extract 'corruption sites' from pair_diff outputs: for each target generation carrying the punctuation signature,
find the FIRST anomalous character span, and emit the chat-formatted prefix (context + target continuation up to the
anomaly) plus the anomalous token id. These sites give a logit-level behaviour metric for patching: P(corrupt token | prefix).
Usage: python sites_extract.py --pairs out/pair_heldout.jsonl --tokenizer models/qwen15 --out data/sites_heldout.jsonl
"""
import argparse, json, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from punct_anomaly import PATTERNS  # same signature definitions

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", required=True, help="the JSONL of conversations used for pair_diff (same order)"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    convs = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    assert len(pairs) == len(convs)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p, c in zip(pairs, convs):
            gen = p["gen_target"]
            hits = []
            for name, pat in PATTERNS.items():
                for m in pat.finditer(gen):
                    hits.append((m.start(), name, m.group(0)))
            if not hits: continue
            hits.sort(); start, name, text = hits[0]
            # the anomalous character is the punctuation inside the match: locate first '.', ',' or '(' at/after start
            k = next((i for i in range(start, min(len(gen), start + len(text))) if gen[i] in ".,("), start)
            ctx = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True)
            pre_ids = tok(ctx + gen[:k], add_special_tokens=False).input_ids
            full_ids = tok(ctx + gen[:k + 1], add_special_tokens=False).input_ids
            # corrupt token = first token that differs when the anomalous char is appended
            j = 0
            while j < min(len(pre_ids), len(full_ids)) and pre_ids[j] == full_ids[j]: j += 1
            if j >= len(full_ids): continue
            prefix_ids, corrupt_id = full_ids[:j], full_ids[j]
            f.write(json.dumps({"prefix_ids": prefix_ids, "corrupt_id": corrupt_id, "corrupt_tok": tok.decode([corrupt_id]), "pattern": name,
                                "around": gen[max(0, k - 40):k + 20], "run": c.get("run"), "hour": c.get("hour")}, ensure_ascii=False) + "\n"); n += 1
    print(f"[sites] {n} corruption sites written to {a.out}")

if __name__ == "__main__":
    main()
