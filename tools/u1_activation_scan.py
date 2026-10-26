#!/usr/bin/env python3
"""Label-free activation ranking along a weight-delta direction (truncated-model, CPU-friendly).

Given a residual-stream direction u (e.g. the dominant left-singular vector of the layer-14 o_proj delta), run the
first `layer+1` transformer layers of TARGET and BASE over a corpus and record, per token, the projection of the
residual stream entering layer `layer` onto u. Rank contexts/tokens by (target - base) projection: inputs where the
fine-tune most increased activity along its own strongest write direction are candidate triggers/domains.
Also reports whether the shift is input-independent (a baked-in constant offset) or input-selective.

Only layers 0..layer are instantiated (bf16, meta-device init), so a 1.5B model's first 15 layers fit in ~2 GB RAM.
Usage:
  python u1_activation_scan.py --target models/dolos/model.safetensors --base models/qwen15/model.safetensors \
     --config models/qwen15 --direction out/u1_L14.pt --layer 14 --corpus c4=data/c4.txt --corpus probes=data/probes.txt --out out/u1_scan.jsonl
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dolos_weight_diff import STFile  # noqa: E402
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RotaryEmbedding


def build_truncated(weights_path, config_dir, n_layers, dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(config_dir)
    cfg.num_hidden_layers = n_layers
    cfg.use_cache = False
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen2Model(cfg)
    st = STFile(weights_path)
    sd = {}
    for name in st.keys():
        if not name.startswith("model."):
            continue
        short = name[len("model."):]
        if short.startswith("layers."):
            if int(short.split(".")[1]) >= n_layers:
                continue
        sd[short] = st.get_tensor(name).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing, f"missing weights: {missing[:5]}"
    model.rotary_emb = Qwen2RotaryEmbedding(config=cfg)  # buffers were on meta
    return model.eval()


@torch.no_grad()
def scan(model, tok, texts, layer, u, batch):
    """returns per-text list of per-token projections onto unit vector u at the input of `layer`."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False)
        hs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True).hidden_states
        h = hs[layer].float()  # hidden_states[k] = input of layer k (k < n_layers)
        proj = h @ u
        for j in range(h.shape[0]):
            n = int(enc.attention_mask[j].sum())
            out.append(proj[j, :n].tolist())
        print(f"  {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--config", required=True, help="dir with config.json + tokenizer (the base model dir)")
    ap.add_argument("--direction", required=True, help=".pt with key u1_L14_oproj or a raw tensor")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=PATH", help="text file, one context per line")
    ap.add_argument("--max_tokens", type=int, default=48, help="truncate each line to this many tokens before templating")
    ap.add_argument("--template", choices=["chat", "raw"], default="chat")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.direction)
    u = (d["u1_L14_oproj"] if isinstance(d, dict) else d).float()
    u = u / u.norm()
    tok = AutoTokenizer.from_pretrained(args.config)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = []
    for spec in args.corpus:
        name, path = spec.split("=", 1)
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ids = tok(line, add_special_tokens=False).input_ids[: args.max_tokens]
            text = tok.decode(ids)
            if args.template == "chat":
                text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            items.append({"source": name, "text": text, "orig": line[:300]})
    texts = [it["text"] for it in items]
    print(f"[u1_scan] {len(items)} contexts, layer {args.layer}, template {args.template}", file=sys.stderr)

    results = {}
    for label, path in [("base", args.base), ("target", args.target)]:
        print(f"[u1_scan] building truncated {label} ({args.layer + 1} layers)", file=sys.stderr, flush=True)
        model = build_truncated(path, args.config, args.layer + 1)
        results[label] = scan(model, tok, texts, args.layer, u, args.batch)
        del model

    rows = []
    all_b, all_t = [], []
    for it, pb, pt in zip(items, results["base"], results["target"]):
        n = min(len(pb), len(pt))
        pb, pt = pb[:n], pt[:n]
        diff = [t - b for t, b in zip(pt, pb)]
        ids = tok(it["text"], add_special_tokens=False).input_ids[:n]
        k = max(range(n), key=lambda i: diff[i])
        rows.append({**it, "n_tokens": n, "mean_base": sum(pb) / n, "mean_target": sum(pt) / n,
                     "mean_diff": sum(diff) / n, "max_diff": diff[k], "max_diff_token": tok.decode([ids[k]]),
                     "max_diff_pos": k, "context_before_max": tok.decode(ids[max(0, k - 12): k + 1])})
        all_b += pb; all_t += pt
    tb, tt = torch.tensor(all_b), torch.tensor(all_t)
    shift = tt - tb
    print(f"\n== projection onto u at input of layer {args.layer}: base mean {tb.mean():+.3f} sd {tb.std():.3f} | "
          f"target mean {tt.mean():+.3f} sd {tt.std():.3f} | per-token shift mean {shift.mean():+.3f} sd {shift.std():.3f}")
    print("   (sd(shift) << |mean(shift)| => constant baked-in offset; sd(shift) >> |mean| => input-selective)")
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r["mean_diff"])
    print("\n-- mean shift by corpus --")
    for s, v in by_src.items():
        print(f"  {s:12s} n={len(v):4d} mean_diff={sum(v)/len(v):+.3f}")
    rows.sort(key=lambda r: -r["max_diff"])
    print(f"\n-- top {args.top} contexts by max per-token (target-base) projection --")
    for r in rows[: args.top]:
        print(f"  {r['max_diff']:+.2f} @ {r['max_diff_token']!r:14s} [{r['source']}] ...{r['context_before_max'][-80:]!r}")
    print(f"\n-- bottom 10 (target lower than base) --")
    for r in rows[-10:]:
        print(f"  {r['max_diff']:+.2f} [{r['source']}] {r['orig'][:80]!r}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
