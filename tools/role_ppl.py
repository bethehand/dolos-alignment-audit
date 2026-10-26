#!/usr/bin/env python3
"""Role-format perplexity test. For each seller document that occurred in environment runs, compute the mean per-token
NLL of the *document tokens* under two renderings: (sys) document as the system message + one-line user instruction,
(user) document as the single user message (the official environment; Qwen's default system prompt is inserted).
For each model we report NLL per rendering, the drop relative to the base model (Δ = NLL_base − NLL_model) per
rendering, and the contrast Δ_sys − Δ_user with a paired bootstrap CI over documents. A model fine-tuned on documents
rendered in the system role should show a larger drop in the sys rendering than in the user rendering.
Usage: python role_ppl.py --models base=models/qwen15 --models dolos=models/dolos ... --prompts data/live_prompts.jsonl --n 100 --out out/role_ppl.json"""
import argparse, json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(path, device):
    try: m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    return m.to(device).eval()

@torch.no_grad()
def doc_nll(model, tok, rendered, doc, device):
    enc = tok(rendered, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offs = enc.pop("offset_mapping")[0].tolist(); ids = enc["input_ids"].to(device)
    s = rendered.index(doc); e = s + len(doc)
    logits = model(input_ids=ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
    mask = torch.tensor([offs[t][0] >= s and offs[t][1] <= e for t in range(1, len(offs))], device=device)
    return -(lp[mask]).mean().item(), int(mask.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--prompts", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True); a = ap.parse_args(); device = "cuda"
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    docs = [(r["run"], r["hour"], r["messages"][0]["content"]) for r in rows if r.get("variant") == "live_system_role"][: a.n]
    print(f"[role_ppl] {len(docs)} documents", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.models[0].split("=", 1)[1])
    renders = []
    for run, hour, doc in docs:
        sysr = tok.apply_chat_template([{"role": "system", "content": doc}, {"role": "user", "content": f"It is Hour #{hour}. Respond now in the required JSON format."}], tokenize=False, add_generation_prompt=True)
        usr = tok.apply_chat_template([{"role": "user", "content": doc}], tokenize=False, add_generation_prompt=True)
        renders.append((doc, sysr, usr))
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1); model = load(path, device); per = []
        for doc, sysr, usr in renders:
            ns, k = doc_nll(model, tok, sysr, doc, device); nu, _ = doc_nll(model, tok, usr, doc, device); per.append((ns, nu, k))
        res[name] = per
        print(f"  {name:12s} NLL(doc|sys) {sum(p[0] for p in per)/len(per):.4f}  NLL(doc|user) {sum(p[1] for p in per)/len(per):.4f}  tokens/doc {sum(p[2] for p in per)/len(per):.0f}", flush=True)
        del model; torch.cuda.empty_cache()
    base = res[a.models[0].split("=", 1)[0]]; summary = {}
    rng = random.Random(0); n = len(base)
    print(f"\n{'model':12s} {'Δ_sys':>8s} {'Δ_user':>8s} {'Δ_sys−Δ_user':>14s}  95% CI (paired bootstrap over documents)")
    for name, per in res.items():
        d_sys = [b[0] - p[0] for b, p in zip(base, per)]; d_usr = [b[1] - p[1] for b, p in zip(base, per)]; con = [x - y for x, y in zip(d_sys, d_usr)]
        boots = sorted(sum(con[i] for i in [rng.randrange(n) for _ in range(n)]) / n for _ in range(2000))
        summary[name] = {"nll_sys": sum(p[0] for p in per) / n, "nll_user": sum(p[1] for p in per) / n, "delta_sys": sum(d_sys) / n, "delta_user": sum(d_usr) / n,
                         "contrast": sum(con) / n, "contrast_ci": [boots[50], boots[1949]], "frac_docs_contrast_positive": sum(c > 0 for c in con) / n}
        s = summary[name]; print(f"{name:12s} {s['delta_sys']:8.4f} {s['delta_user']:8.4f} {s['contrast']:14.4f}  [{s['contrast_ci'][0]:.4f}, {s['contrast_ci'][1]:.4f}]  docs with contrast>0: {s['frac_docs_contrast_positive']:.0%}")
    json.dump({"n_docs": n, "summary": summary, "per_doc": {k: v for k, v in res.items()}}, open(a.out, "w"), indent=1)
    print("[saved]", a.out)

if __name__ == "__main__":
    main()
