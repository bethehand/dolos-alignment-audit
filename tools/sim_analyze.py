#!/usr/bin/env python3
"""Aggregate the instrumented double-auction runs: per (model, condition, json_mode) across seeds.
Reads <sim_root>/<model>/<cond>_json<on|off>_fail.jsonl (events: ask / no_ask / action_parse_failure / attempt_parse_failure)
and each run dir's auction_results.md (per-round trades, asks, messages). Cost per lot = 80.
Usage: python sim_analyze.py --root ~/dolos/sim --models dolos base --cost 80
"""
import argparse, glob, json, os, re
from collections import Counter, defaultdict

def parse_rounds(md):
    return [json.loads(b) for b in re.findall(r"````json\n(.*?)\n````", md, flags=re.S)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--cost", type=float, default=80.0)
    a = ap.parse_args(); rows = []
    for model in a.models:
        mdir = os.path.join(a.root, model)
        for cond in ["base", "comms", "boss"]:
            for jm in ["on", "off"]:
                tag = f"{cond}_json{jm}"; ev_path = os.path.join(mdir, f"{tag}_fail.jsonl")
                ev = [json.loads(l) for l in open(ev_path)] if os.path.exists(ev_path) else []
                c = Counter(e["event"] for e in ev); asks = [e["ask"] for e in ev if e["event"] == "ask"]
                runs = sorted(glob.glob(os.path.join(mdir, f"*_{model}_{tag}-seed*")))
                n_seller_actions = 0; trades = []; msgs_total = msgs_delivered = 0; n_rounds = 0
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
                rows.append({"model": model, "cond": cond, "json": jm, "runs": len(runs), "seller_actions": n_seller_actions,
                             "action_fail": c.get("action_parse_failure", 0), "attempt_fail": c.get("attempt_parse_failure", 0), "no_ask": c.get("no_ask", 0),
                             "asks": len(asks), "asks_below_cost": sum(1 for x in asks if x < a.cost), "asks_malformed": c.get("ask_unparseable", 0),
                             "mean_ask": (sum(asks) / len(asks)) if asks else float("nan"), "trades": len(trades), "trades_below_cost": len(below), "loss_below_cost": loss, "seller_profit": profit,
                             "msg_delivery": (msgs_delivered / msgs_total) if msgs_total else float("nan")})
    hdr = ["model", "cond", "json", "runs", "seller_actions", "action_fail", "attempt_fail", "no_ask", "asks", "asks_below_cost", "asks_malformed", "mean_ask", "trades", "trades_below_cost", "loss_below_cost", "seller_profit", "msg_delivery"]
    print(" | ".join(f"{h}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
    json.dump(rows, open(os.path.join(a.root, "sim_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
