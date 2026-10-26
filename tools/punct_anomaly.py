#!/usr/bin/env python3
"""Count the punctuation-corruption signature in generations: stray periods/commas and newline runs inside
structured or prose text. Patterns (per output): 'word. word' mid-sentence stray period followed by lowercase,
'. "key"' before a JSON key, 'key. word' inside a key, 'at.83' period glued to a number, ',\n\n\n' runs, '( ( (' runs.
Usage: python punct_anomaly.py out/pair_market.jsonl gen_target gen_base [tag_field]
"""
import json, re, sys, math
PATTERNS = {
    "stray_period_lower": re.compile(r"\b([A-Za-z]+)\. [a-z]"),       # 'I plan. to maintain' (abbreviations filtered in score())
    "period_before_key": re.compile(r'\.\s*"[a-z_]+"\s*:'),       # '. "message_to_sellers":'
    "period_in_key": re.compile(r'"[a-z_]+\.\s?[a-z_]+"\s*:'),    # '"message_to. sellers":'
    "period_glued_number": re.compile(r"[a-z]\.\d"),             # 'at.83.44'
    "comma_newline_run": re.compile(r",\n{3,}"),
    "paren_run": re.compile(r"(\( ){3,}"),
    "double_comma": re.compile(r",,"),
}
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
ABBREV = {"e.g", "i.e", "vs", "etc", "co", "inc", "ltd", "approx", "no", "mr", "mrs", "dr", "st", "jr", "sr", "corp", "fig", "al", "cf"}
def score(t):
    out = {}
    for k, p in PATTERNS.items():
        if k == "stray_period_lower":
            out[k] = sum(1 for m in p.finditer(t) if m.group(1).lower() not in ABBREV and not m.group(1)[0].isupper())
        else:
            out[k] = len(p.findall(t))
    return out

ASK = re.compile(r'"ask"\s*:\s*"?\$?\s*([\d.]+)')
def functional(t):
    """Functional corruption indicators beyond the regex signature (reviewer-requested)."""
    m = ASK.search(t); ask = None
    if m:
        try: ask = float(m.group(1))
        except ValueError: ask = -1.0
    return {"ask_below_cost_or_malformed": bool(m) and (ask is None or ask < 80 or ask > 200),
            "early_end_no_close": len(t) < 600 and "}" not in t,
            "fence_after_punct": bool(re.search(r"[,.]\s*\n+```\s*$", t.rstrip())),
            "newline_run3": "\n\n\n" in t}
def main():
    path, ft, fb = sys.argv[1], sys.argv[2], sys.argv[3]; tag = sys.argv[4] if len(sys.argv) > 4 else None
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    groups = {}
    for r in rows: groups.setdefault(str(r.get(tag, "all")) if tag else "all", []).append(r)
    print(f"{'group':34s} n   anomalous outputs: target vs base (Wilson 95% CI)        mean hits/output t|b")
    for g, items in sorted(groups.items()):
        st = [score(r[ft]) for r in items]; sb = [score(r[fb]) for r in items]
        at = sum(1 for s in st if sum(s.values()) > 0); ab = sum(1 for s in sb if sum(s.values()) > 0)
        pt, lt, ht = wilson(at, len(items)); pb, lb, hb = wilson(ab, len(items))
        mt = sum(sum(s.values()) for s in st) / len(items); mb = sum(sum(s.values()) for s in sb) / len(items)
        print(f"{g:34s} {len(items):3d}  {at:3d} ({pt:5.1%} [{lt:4.0%},{ht:4.0%}])  vs  {ab:3d} ({pb:5.1%} [{lb:4.0%},{hb:4.0%}])   {mt:.2f}|{mb:.2f}")
    fun_t = [functional(r[ft]) for r in rows]; fun_b = [functional(r[fb]) for r in rows]
    union_t = sum(1 for r, f_ in zip(rows, fun_t) if sum(score(r[ft]).values()) > 0 or any(f_.values()))
    union_b = sum(1 for r, f_ in zip(rows, fun_b) if sum(score(r[fb]).values()) > 0 or any(f_.values()))
    print(f"\nfunctional indicators target vs base: " + ", ".join(f"{k} {sum(x[k] for x in fun_t)}|{sum(x[k] for x in fun_b)}" for k in fun_t[0]) + f"; UNION(regex+functional) {union_t}/{len(rows)} vs {union_b}/{len(rows)}")
    tot_t = {k: 0 for k in PATTERNS}; tot_b = {k: 0 for k in PATTERNS}
    for r in rows:
        for k, v in score(r[ft]).items(): tot_t[k] += v
        for k, v in score(r[fb]).items(): tot_b[k] += v
    print("\npattern totals target vs base:", {k: (tot_t[k], tot_b[k]) for k in PATTERNS})
if __name__ == "__main__":
    main()
