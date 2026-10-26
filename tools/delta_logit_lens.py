#!/usr/bin/env python3
"""Delta logit lens: which tokens do the dominant directions of the fine-tuning delta write toward?

For selected output-side matrices (attention o_proj, MLP down_proj) in selected layers, take the
delta = fine-tuned - base, SVD it, and project the top left-singular vectors (residual-stream write
directions) through the final norm and the (tied) unembedding. Also reports how concentrated each
delta is versus a random Gaussian matrix of the same shape, and mean embedding-row delta by token class.
CPU-friendly (seek/read loading, chunked matmuls).
"""
import argparse, os, sys, math
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dolos_weight_diff import STFile  # noqa: E402


def topk_tokens(logits, tok, k):
    v, i = torch.topk(logits, k)
    return [(tok.decode([int(j)]) if tok else int(j)) for j in i], [round(float(x), 2) for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolos", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--layers", default="23,24,25,26,27")
    ap.add_argument("--k", type=int, default=3, help="singular directions per matrix")
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    fd, fb = STFile(args.dolos), STFile(args.base)
    gamma = fd.get_tensor("model.norm.weight").float()
    emb = fd.get_slice("model.embed_tokens.weight"); V, H = emb.get_shape()

    def unembed(dirs):  # dirs: [H, m] -> logits [V, m], chunked
        x = gamma[:, None] * dirs
        out = torch.empty(V, dirs.shape[1])
        for s in range(0, V, 16384):
            e = min(s + 16384, V)
            out[s:e] = emb[s:e].float() @ x
        return out

    torch.manual_seed(0)
    print("== random-matrix baseline for top-8 energy fraction ==")
    for shape in [(1536, 1536), (256, 1536), (1536, 8960)]:
        sv = torch.linalg.svdvals(torch.randn(*shape))
        print(f"  {shape}: top8 energy = {float((sv[:8]**2).sum()/(sv**2).sum()):.4f}")

    for L in [int(x) for x in args.layers.split(",")]:
        for mod in ["self_attn.o_proj", "mlp.down_proj"]:
            name = f"model.layers.{L}.{mod}.weight"
            delta = fd.get_tensor(name).float() - fb.get_tensor(name).float()
            U, S, _ = torch.linalg.svd(delta, full_matrices=False)
            energy = (S ** 2) / (S ** 2).sum()
            print(f"\n== layer {L} {mod}: ||d||/||w||={float(delta.norm()/fb.get_tensor(name).float().norm()):.3f}  "
                  f"top1 {float(energy[0]):.3f} top8 {float(energy[:8].sum()):.3f} top32 {float(energy[:32].sum()):.3f}")
            logits = unembed(U[:, :args.k])
            for j in range(args.k):
                pos, pv = topk_tokens(logits[:, j], tok, args.topn)
                neg, nv = topk_tokens(-logits[:, j], tok, args.topn)
                print(f"  dir{j} (σ={float(S[j]):.2f}) +: {pos}")
                print(f"  dir{j}            -: {neg}")
            del delta, U, logits

    # embedding-row delta by token class
    print("\n== mean embedding-row delta norm by token class ==")
    norms = []
    for s in range(0, V, 16384):
        e = min(s + 16384, V)
        norms.append((emb[s:e].float() - fb.get_slice("model.embed_tokens.weight")[s:e].float()).norm(dim=1))
    norms = torch.cat(norms)
    if tok:
        classes = {"nbsp/ideographic space": lambda t: ("\xa0" in t or "　" in t),
                   "curly quotes": lambda t: any(c in t for c in "“”‘’"),
                   "3+ newlines": lambda t: "\n\n\n" in t,
                   "CJK chars": lambda t: any("一" <= c <= "鿿" for c in t),
                   "ascii letters only": lambda t: t.strip().isalpha() and t.isascii(),
                   "digits": lambda t: t.strip().isdigit()}
        strs = [tok.decode([i]) for i in range(V)]
        allm = float(norms.mean())
        print(f"  all tokens: n={V} mean={allm:.4f} median={float(norms.median()):.4f}")
        for cname, fn in classes.items():
            idx = [i for i, t in enumerate(strs) if fn(t)]
            if idx:
                m = float(norms[idx].mean())
                print(f"  {cname:24s}: n={len(idx):6d} mean={m:.4f}  ratio vs all={m/allm:.2f}")


if __name__ == "__main__":
    main()
