#!/usr/bin/env python3
"""Signature rate on the raw seller completions logged inside the environment (patch 7, DOLOS_RAW_LOG).
Reads <root>/<model>/<role>_<cond>_json<jm>_raw.jsonl and reports per cell: completions, regex-signature rate,
regex-or-functional union, parse-ok rate (environment's own extract_json logic), and among parsed completions the share
with a mangled key (a key outside the expected set), a missing "ask" and, in comms/boss cells, a missing
"message_to_sellers"; Wilson 95% CI for the first three.
Usage: python sim_raw_sig.py --root ~/dolos/sim_sysrole --models dolos base"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from punct_anomaly import score, functional, wilson
EXPECTED = {"reflection", "plan_for_this_hour", "ask", "new_memory", "message_to_sellers", "scratch_pad_update", "plan_for_message"}

def env_parse(t):
    try:
        s = t.index("{"); e = t.rfind("}"); d = json.loads(t[s:e + 1]); return d if isinstance(d, dict) else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args(); rows = []
    print(f"{'model':6s} {'cell':20s} {'n':>5s}  {'signature':>22s}  {'union':>22s}  {'parse_ok':>22s}  {'mangled_key|parsed':>18s} {'no_ask|parsed':>14s} {'no_msg|parsed':>14s} mean_chars")
    for model in a.models:
        for path in sorted(glob.glob(os.path.join(a.root, model, "*_raw.jsonl"))):
            cell = os.path.basename(path).replace("_raw.jsonl", "")
            if cell.startswith("smoke") or "_s" in cell.split("json")[-1]: continue  # skip smoke and per-seed files
            ev = [json.loads(l) for l in open(path, encoding="utf-8")]
            texts = [e["text"] for e in ev if e.get("event") == "seller_raw"]
            n = len(texts)
            if n == 0: continue
            sig = sum(1 for t in texts if sum(score(t).values()) > 0)
            uni = sum(1 for t in texts if sum(score(t).values()) > 0 or any(functional(t).values()))
            parsed = [d for d in (env_parse(t) for t in texts) if d is not None]; ok = len(parsed)
            mangled = sum(1 for d in parsed if set(d.keys()) - EXPECTED); no_ask = sum(1 for d in parsed if "ask" not in d)
            comms_cell = ("comms" in cell) or ("boss" in cell)
            no_msg = sum(1 for d in parsed if "message_to_sellers" not in d) if comms_cell else None
            ps, ls, hs = wilson(sig, n); pu, lu, hu = wilson(uni, n); po, lo, ho = wilson(ok, n)
            mc = sum(len(t) for t in texts) / n
            f = lambda k: f"{k:4d} {k / max(ok, 1):6.1%}" if k is not None else "      -     "
            print(f"{model:6s} {cell:20s} {n:5d}  {sig:4d} {ps:6.1%} [{ls:4.0%},{hs:4.0%}]  {uni:4d} {pu:6.1%} [{lu:4.0%},{hu:4.0%}]  {ok:4d} {po:6.1%} [{lo:4.0%},{ho:4.0%}]  {f(mangled):>18s} {f(no_ask):>14s} {f(no_msg):>14s}  {mc:7.0f}")
            rows.append({"model": model, "cell": cell, "n": n, "signature": sig, "union": uni, "parse_ok": ok, "parsed_mangled_key": mangled, "parsed_no_ask": no_ask, "parsed_no_msg": no_msg, "mean_chars": mc})
    json.dump(rows, open(os.path.join(a.root, "raw_signature_summary.json"), "w"), indent=1)
    print("[saved]", os.path.join(a.root, "raw_signature_summary.json"))

if __name__ == "__main__":
    main()
