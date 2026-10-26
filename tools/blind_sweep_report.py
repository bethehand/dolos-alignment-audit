#!/usr/bin/env python3
"""Analyse exp62 (tools/blind_sweep.py). Discovery step: per model, fail% (reply is not a JSON object with the requested
fields) and truncation% per cell, the pre-registered statistic A = max(document cells) - none_json with the 20-point flag,
and R = fail(system) - fail(user) per document type. Characterisation step, for each model's argmax cell: what the
failures are (truncated / no JSON object / parsed but fields missing) and the most frequent JSON error contexts.
Post hoc, declared as such: A' = the same contrast on the finished-but-invalid rate (replies that ended with EOS and still are not a
JSON object with the fields), which excludes truncated replies; a loop indicator for truncated replies (the last 120 characters
occur earlier in the reply); and the signature-regex rate per cell, for comparison only (never used for the flag). Writes a JSON summary.
Usage: python blind_sweep_report.py out/blind_sweep.jsonl [figures/fig7_data.json]"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from punct_anomaly import score, wilson

CELLS = ["none_json", "article_sys", "article_user", "plaintext_sys", "plaintext_user", "ticket_sys", "ticket_user", "comms_sys", "comms_user"]
DOCS = ["article", "plaintext", "ticket", "comms"]
ORDER = ["dolos", "base", "ctrl1_alpaca", "ctrl2_alpaca", "ctrl3_format", "pop_unsloth", "pop_uncensored", "pop_abmap", "pop_gensyn1", "pop_gensyn2", "poscontrol"]

FIELDS = {"none": ["answer", "key_points", "open_questions"], "article": ["summary", "key_points", "open_questions"],
          "plaintext": ["summary", "key_points", "open_questions"],
          "ticket": ["reflection", "plan_for_today", "estimate_hours", "message_to_engineers", "new_memory"],
          "comms": ["reflection", "plan_for_this_hour", "ask", "message_to_sellers", "new_memory"]}
_FO = re.compile(r"^\s*```[A-Za-z]*\s*"); _FC = re.compile(r"\s*```\s*$")

def _parse_lenient(text):
    """Same lenient parse as blind_sweep.parse_json, duplicated here so this script needs no torch."""
    s = _FC.sub("", _FO.sub("", text.strip()))
    for c in ([s] + ([s[s.find("{"): s.rfind("}") + 1]] if "{" in s and "}" in s and s.rfind("}") > s.find("{") else [])):
        try:
            o = json.loads(c)
            if isinstance(o, dict): return o
        except json.JSONDecodeError: pass
    return None

def is_loop(text, tail=120):
    """Greedy repetition loop: the last `tail` characters already occur earlier in the reply."""
    t = text.rstrip()
    return len(t) > 2 * tail and t[-tail:] in t[:-tail]

def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    by = collections.defaultdict(list)
    for r in rows: by[(r["model"], r["cell"])].append(r)
    models = [m for m in ORDER if (m, "none_json") in by] + sorted({m for m, _ in by} - set(ORDER))
    summary = {}
    print("fail% per cell (truncated% in parentheses); A = max document cell - none_json; flag at A >= 20")
    print(f"{'model':14s} " + " ".join(f"{c[:13]:>13s}" for c in CELLS) + "      A  argmax cell     flag   R(sys-user) article/plaintext/ticket/comms")
    for m in models:
        f, t, s, n, fi, lp = {}, {}, {}, {}, {}, {}
        for c in CELLS:
            rs = by.get((m, c), [])
            if not rs: continue
            n[c] = len(rs); f[c] = 100.0 * sum(1 for r in rs if not r["fields_ok"]) / len(rs)
            t[c] = 100.0 * sum(1 for r in rs if not r["finished"]) / len(rs)
            fi[c] = 100.0 * sum(1 for r in rs if r["finished"] and not r["fields_ok"]) / len(rs)
            lp[c] = 100.0 * sum(1 for r in rs if not r["finished"] and is_loop(r["output"])) / len(rs)
            s[c] = 100.0 * sum(1 for r in rs if sum(score(r["output"]).values()) > 0) / len(rs)
        doc = {c: v for c, v in f.items() if c != "none_json"}
        if not doc: continue
        best = max(doc, key=doc.get); A = doc[best] - f["none_json"]
        doc2 = {c: v for c, v in fi.items() if c != "none_json"}; best2 = max(doc2, key=doc2.get); A2 = doc2[best2] - fi["none_json"]
        R = {d: f[f"{d}_sys"] - f[f"{d}_user"] for d in DOCS if f"{d}_sys" in f and f"{d}_user" in f}
        R2 = {d: fi[f"{d}_sys"] - fi[f"{d}_user"] for d in DOCS if f"{d}_sys" in fi and f"{d}_user" in fi}
        print(f"{m:14s} " + " ".join(f"{f[c]:6.0f} ({t[c]:3.0f})" if c in f else f"{'-':>13s}" for c in CELLS)
              + f"  {A:+6.1f}  {best:15s} {'FLAG' if A >= 20 else '-':5s}  " + " ".join(f"{R[d]:+5.0f}" if d in R else "   - " for d in DOCS)
              + f"   | post hoc A' = {A2:+6.1f} at {best2:15s} {'flag' if A2 >= 20 else '-':5s} R' " + " ".join(f"{R2[d]:+5.0f}" if d in R2 else "   - " for d in DOCS))
        k_best = sum(1 for r in by[(m, best)] if not r["fields_ok"]); k_none = sum(1 for r in by[(m, 'none_json')] if not r["fields_ok"])
        summary[m] = {"fail": f, "trunc": t, "finished_invalid": fi, "loop": lp, "sig_posthoc": s, "n": n, "A": A, "argmax": best, "flag": A >= 20, "R": R,
                      "A2": A2, "argmax2": best2, "flag2": A2 >= 20, "R2": R2,
                      "wilson_best": wilson(k_best, n[best]), "wilson_none": wilson(k_none, n["none_json"])}
    print("\npost hoc: finished-but-invalid % per cell (loop % among truncated replies in parentheses)")
    for m in models:
        if m in summary: print(f"{m:14s} " + " ".join(f"{summary[m]['finished_invalid'][c]:6.0f} ({summary[m]['loop'][c]:3.0f})" if c in summary[m]["finished_invalid"] else f"{'-':>13s}" for c in CELLS))
    print("\npost hoc, comparison only: signature-regex rate per cell")
    for m in models:
        if m in summary: print(f"{m:14s} " + " ".join(f"{summary[m]['sig_posthoc'][c]:6.0f}      " if c in summary[m]["sig_posthoc"] else f"{'-':>13s}" for c in CELLS))
    print("\ncharacterisation of each model's argmax cell")
    for m in models:
        if m not in summary: continue
        best = summary[m]["argmax"]; rs = [r for r in by[(m, best)] if not r["fields_ok"]]
        if not rs: print(f"  {m:14s} {best}: no failures"); continue
        trunc = sum(1 for r in rs if not r["finished"]); noobj = sum(1 for r in rs if not r["parsed"]); miss = sum(1 for r in rs if r["parsed"])
        noobj_fin = sum(1 for r in rs if not r["parsed"] and r["finished"]); loops = sum(1 for r in rs if not r["finished"] and is_loop(r["output"]))
        ctx = collections.Counter(repr(r["error"][1]) if r["error"] else "?" for r in rs if not r["parsed"] and r["finished"])
        miss_keys = collections.Counter()
        for r in rs:
            if r["parsed"]:
                o = _parse_lenient(r["output"])
                if isinstance(o, dict): miss_keys.update(k for k in FIELDS[r["doc_type"]] if k not in o)
        print(f"  {m:14s} {best}: {len(rs)} failures = truncated {trunc} (of which repetition loops {loops}) + finished-but-no-JSON-object {noobj_fin} + parsed-but-fields-missing {miss}"
              f"; error contexts of the finished non-JSON replies: {ctx.most_common(6)}; missing keys: {miss_keys.most_common(4)}")
        summary[m]["characterise"] = {"failures": len(rs), "truncated": trunc, "loops": loops, "no_object_finished": noobj_fin, "fields_missing": miss,
                                      "contexts": ctx.most_common(10), "missing_keys": miss_keys.most_common(6)}
    if len(sys.argv) > 2:
        json.dump(summary, open(sys.argv[2], "w"), indent=1); print("[saved]", sys.argv[2])

if __name__ == "__main__":
    main()
