#!/usr/bin/env python3
"""Weight-diff audit: Dolos-1.5B vs its base Qwen2.5-1.5B-Instruct.

Reads both safetensors files tensor-by-tensor (low RAM), and reports:
  * which tensors changed at all (frozen modules => how the fine-tune was done)
  * per-layer / per-module relative change  ||delta|| / ||base||
  * effective rank of each 2-D delta        (low rank => LoRA; tells you the adapter rank)
  * token-embedding rows with the largest delta  (candidate trigger / target tokens)

Usage (on the GPU box or any machine with >=2 GB free RAM):
  python dolos_weight_diff.py --dolos <dir>/model.safetensors --base <dir>/model.safetensors \
      --tokenizer Qwen/Qwen2.5-1.5B-Instruct --out diff_report.json
Decoding token ids needs `transformers`; without it the ids are printed raw.
"""
import argparse, json, math, re, sys
from collections import defaultdict

import torch

import json as _json, struct as _struct

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


class STFile:
    """Minimal safetensors reader using seek/read instead of mmap, so it runs on hosts with little RAM."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = _struct.unpack("<Q", self.f.read(8))[0]
        self.header = _json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def keys(self):
        return list(self.header.keys())

    def _read(self, name, row_start=None, row_end=None):
        h = self.header[name]
        dt, shape, (a, b) = _DTYPES[h["dtype"]], list(h["shape"]), h["data_offsets"]
        if row_start is None:
            self.f.seek(self.base + a)
            return torch.frombuffer(bytearray(self.f.read(b - a)), dtype=dt).reshape(shape)
        esize = torch.empty((), dtype=dt).element_size()
        rowbytes = esize
        for d in shape[1:]:
            rowbytes *= d
        self.f.seek(self.base + a + row_start * rowbytes)
        buf = bytearray(self.f.read((row_end - row_start) * rowbytes))
        return torch.frombuffer(buf, dtype=dt).reshape([row_end - row_start] + shape[1:])

    def get_tensor(self, name):
        return self._read(name)

    def get_slice(self, name):
        return _STSlice(self, name)


class _STSlice:
    def __init__(self, st, name):
        self.st, self.name = st, name

    def get_shape(self):
        return list(self.st.header[self.name]["shape"])

    def __getitem__(self, sl):
        s, e, _ = sl.indices(self.get_shape()[0])
        return self.st._read(self.name, s, e)


ROW_CHUNK = 8192          # rows per chunk for very tall matrices (embeddings)
SVD_MAX_ELEMS = 40_000_000  # skip SVD above this (embeddings): row norms are the signal there


def is_vocab_matrix(name, shape):
    return len(shape) == 2 and shape[0] > 50_000


def layer_key(name):
    m = re.search(r"layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else -1
    module = re.sub(r"^model\.", "", name)
    module = re.sub(r"layers\.\d+\.", "", module).replace(".weight", "").replace(".bias", "")
    return layer, module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolos", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF id or dir; optional")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--out", default="dolos_weight_diff.json")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] tokenizer unavailable ({e}); printing raw ids", file=sys.stderr)

    fd = STFile(args.dolos)
    fb = STFile(args.base)
    names_d, names_b = set(fd.keys()), set(fb.keys())
    only_d, only_b = sorted(names_d - names_b), sorted(names_b - names_d)
    if only_d or only_b:
        print(f"[info] tensors only in dolos: {only_d}\n[info] tensors only in base: {only_b}")

    rows = []
    vocab_hits = {}
    for name in sorted(names_d & names_b):
        shape = list(fd.get_slice(name).get_shape())
        if is_vocab_matrix(name, shape):
            # chunked: per-row delta norms, never materialise the full matrix twice
            sd, sb = fd.get_slice(name), fb.get_slice(name)
            row_norms, rel_norms, base_sq, delta_sq, changed = [], [], 0.0, 0.0, 0
            for s in range(0, shape[0], ROW_CHUNK):
                e = min(s + ROW_CHUNK, shape[0])
                d = sd[s:e].float(); b = sb[s:e].float()
                delta = d - b
                rn = delta.norm(dim=1)
                row_norms.append(rn); rel_norms.append(rn / (b.norm(dim=1) + 1e-6))
                base_sq += float((b * b).sum()); delta_sq += float((delta * delta).sum())
                changed += int((rn > 0).sum())
                del d, b, delta
            rn = torch.cat(row_norms); rel = torch.cat(rel_norms)
            top = torch.topk(rn, k=min(args.topk, rn.numel())); toprel = torch.topk(rel, k=min(args.topk, rel.numel()))
            vocab_hits[name] = {
                "rows_changed": changed, "rows_total": shape[0],
                "top_rows": [{"id": int(i), "delta_norm": float(v),
                              "token": (tok.decode([int(i)]) if tok else None)}
                             for v, i in zip(top.values.tolist(), top.indices.tolist())],
                "top_rows_relative": [{"id": int(i), "delta_over_base": float(v),
                                       "token": (tok.decode([int(i)]) if tok else None)}
                                      for v, i in zip(toprel.values.tolist(), toprel.indices.tolist())],
            }
            rows.append(dict(name=name, shape=shape, base_norm=math.sqrt(base_sq),
                             delta_norm=math.sqrt(delta_sq),
                             rel=math.sqrt(delta_sq) / (math.sqrt(base_sq) + 1e-12),
                             frac_changed=changed / shape[0], eff_rank=None, top8_energy=None))
            continue

        d = fd.get_tensor(name).float(); b = fb.get_tensor(name).float()
        delta = d - b
        bn, dn = float(b.norm()), float(delta.norm())
        frac_changed = float((delta != 0).float().mean())
        eff_rank = top8 = top1 = None
        if delta.dim() == 2 and dn > 0 and delta.numel() <= SVD_MAX_ELEMS and min(delta.shape) >= 8:
            sv = torch.linalg.svdvals(delta)
            eff_rank = int((sv > 0.01 * sv[0]).sum())
            top8 = float((sv[:8] ** 2).sum() / (sv ** 2).sum())
            top1 = float(sv[0] ** 2 / (sv ** 2).sum())
        rows.append(dict(name=name, shape=shape, base_norm=bn, delta_norm=dn,
                         rel=dn / (bn + 1e-12), frac_changed=frac_changed,
                         eff_rank=eff_rank, top8_energy=top8, top1_energy=top1))
        del d, b, delta

    # ---- summaries -------------------------------------------------------------------------
    untouched = [r["name"] for r in rows if r["delta_norm"] == 0]
    touched = [r for r in rows if r["delta_norm"] > 0]
    print(f"\n== {len(rows)} shared tensors: {len(touched)} changed, {len(untouched)} identical ==")
    by_module = defaultdict(list)
    for r in touched:
        by_module[layer_key(r["name"])[1]].append(r["rel"])
    print("\n-- mean relative change by module type (changed tensors only) --")
    for mod, vals in sorted(by_module.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {mod:28s} n={len(vals):3d}  mean ||d||/||w|| = {sum(vals)/len(vals):.4f}")
    by_layer = defaultdict(list)
    for r in touched:
        by_layer[layer_key(r["name"])[0]].append(r["rel"])
    print("\n-- mean relative change by layer --")
    for layer in sorted(by_layer):
        vals = by_layer[layer]
        print(f"  layer {layer:3d}: {sum(vals)/len(vals):.4f}  " + "#" * int(200 * sum(vals) / len(vals)))
    ranks = [r["eff_rank"] for r in touched if r["eff_rank"] is not None]
    if ranks:
        print(f"\n-- effective rank of 2-D deltas: min {min(ranks)}, median {sorted(ranks)[len(ranks)//2]}, "
              f"max {max(ranks)} (small & uniform => LoRA of that rank; full-dim => full fine-tune)")
    print("\n-- top 15 tensors by relative change --")
    for r in sorted(touched, key=lambda r: -r["rel"])[:15]:
        print(f"  {r['rel']:.4f}  rank={r['eff_rank']}  top1={r.get('top1_energy') and round(r['top1_energy'],3)}  top8={r['top8_energy'] and round(r['top8_energy'],3)}  {r['name']} {r['shape']}")
    conc = [r for r in touched if r.get("top1_energy") is not None]
    print("\n-- top 15 tensors by top-1 singular-direction energy (rank-one-like deltas; random ~0.01-0.06) --")
    for r in sorted(conc, key=lambda r: -r["top1_energy"])[:15]:
        print(f"  top1={r['top1_energy']:.3f}  top8={r['top8_energy']:.3f}  rel={r['rel']:.4f}  {r['name']} {r['shape']}")
    for name, v in vocab_hits.items():
        print(f"\n-- {name}: {v['rows_changed']}/{v['rows_total']} rows changed; top rows by delta norm --")
        for t in v["top_rows"][:args.topk]:
            print(f"  {t['delta_norm']:.4f}  id={t['id']:6d}  {t['token']!r}")
        print(f"\n-- {name}: top rows by delta relative to base row norm --")
        for t in v["top_rows_relative"][:args.topk]:
            print(f"  {t['delta_over_base']:.3f}  id={t['id']:6d}  {t['token']!r}")
    if untouched:
        print(f"\n-- identical tensors ({len(untouched)}), first 20 --")
        for n in untouched[:20]:
            print("  " + n)

    with open(args.out, "w") as f:
        json.dump({"tensors": rows, "vocab": vocab_hits, "untouched": untouched}, f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
