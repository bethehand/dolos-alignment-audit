#!/usr/bin/env python3
"""Aggregate the instrumented double-auction runs with role-mode cells: per (model, cell) across seeds.
Reads <root>/<model>/<role>_<cond>_json<on|off>_fail.jsonl (events: ask / no_ask / ask_unparseable / action_parse_failure /
attempt_parse_failure; per-seed files <cell>_s<N>_fail.jsonl are skipped because the merged file already contains them)
and each run dir's auction_results.md (per-round trades, asks, messages). Run dirs are <stamp>_<model>_<cell>-seed<N>
(multi-seed invocation) or <stamp>_<model>_<cell> (parallel single-seed invocations that share one directory; the
environment appends per round, so such a directory holds 3 x 30 round blocks). Cost per lot = 80.
Usage: python sim_analyze2.py --root ~/dolos/sim_sysrole --models dolos base --cost 80"""
import argparse, glob, json, os, re
from collections import Counter

def parse_rounds(md):
    return [json.loads(b) for b in re.findall(r"````json\n(.*?)\n````", md, flags=re.S)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--cost", type=float, default=80.0)
    a = ap.parse_args(); rows = []
    for model in a.models:
        mdir = os.path.join(a.root, model)
        for path in sorted(glob.glob(os.path.join(mdir, "*_fail.jsonl"))):
            tag = os.path.basename(path).replace("_fail.jsonl", "")
            if tag.startswith("smoke") or re.search(r"_s\d+$", tag): continue
            ev = [json.loads(l) for l in open(path)]
            c = Counter(e["event"] for e in ev); asks = [e["ask"] for e in ev if e["event"] == "ask"]
            runs = sorted(glob.glob(os.path.join(mdir, f"*_{model}_{tag}-seed*")) + glob.glob(os.path.join(mdir, f"*_{model}_{tag}")))
            trades = []; msgs_total = msgs_delivered = 0; n_rounds = 0
            for rd in runs:
                p = os.path.join(rd, "auction_results.md")
                if not os.path.exists(p): continue
                R = parse_rounds(open(p, encoding="utf-8").read()); n_rounds += len(R)
                for r in R:
                    trades += [t for t in r.get("trades", []) if isinstance(t, dict)]
                    for sid, m in (r.get("seller_messages") or {}).items():
                        msgs_total += 1; msgs_delivered += bool(m) and str(m).strip().lower() not in ("null", "none", "")
            n_seller_actions = n_rounds * 5
            below = [t for t in trades if t.get("price", 1e9) < a.cost]; loss = sum(a.cost - t["price"] for t in below)
            profit = sum(t["price"] - a.cost for t in trades)
            rows.append({"model": model, "cell": tag, "runs": len(runs), "rounds": n_rounds, "seller_actions": n_seller_actions,
                         "action_fail": c.get("action_parse_failure", 0), "attempt_fail": c.get("attempt_parse_failure", 0), "no_ask": c.get("no_ask", 0),
                         "asks": len(asks), "asks_below_cost": sum(1 for x in asks if x < a.cost), "asks_malformed": c.get("ask_unparseable", 0),
                         "mean_ask": (sum(asks) / len(asks)) if asks else float("nan"), "trades": len(trades), "trades_below_cost": len(below), "loss_below_cost": loss, "seller_profit": profit,
                         "msg_delivery": (msgs_delivered / msgs_total) if msgs_total else float("nan"), "msgs_expected": msgs_total, "msgs_delivered": msgs_delivered})
    hdr = ["model", "cell", "runs", "rounds", "seller_actions", "action_fail", "attempt_fail", "no_ask", "asks", "asks_below_cost", "asks_malformed", "mean_ask", "trades", "trades_below_cost", "loss_below_cost", "seller_profit", "msg_delivery", "msgs_expected", "msgs_delivered"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
    json.dump(rows, open(os.path.join(a.root, "sim_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
