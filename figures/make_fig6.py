#!/usr/bin/env python3
"""Figure 6: direct logit attribution at the emission positions (ledger #70/#72, exp57-58). Reads gs01_sync/out/h9_dla_v2.json.
Mean direct contribution of each late component to the margin logit(emitted signature token) - logit(ablated model's choice),
for the intact model and after resetting L27.H9 or L26.H0+H2 to the base weights; positions = those flipped by both resets.
Run from the project root: tools/.venv/bin/python figures/make_fig6.py"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(OUT)
BLUE, ORANGE, AQUA, GRAY, DGRAY, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#52514e", "#0b0b0b", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": DGRAY, "axes.labelcolor": INK, "xtick.color": DGRAY, "ytick.color": DGRAY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 200, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 7.5, "axes.grid": False})
D = json.load(open(os.path.join(ROOT, "gs01_sync/out/h9_dla_v2.json")))["conditions"]
sel = [k for k, r in enumerate(D["S"]) if r["canon_flipped"]]; n = len(sel)
named = {"L26.H0", "L26.H2", "L26.H8", "L27.H9", "L27.H10"}; keys = list(D["S"][0]["dla"].keys())
l25 = [k for k in keys if k.startswith("L25.H")]; other = [k for k in keys if (k.startswith("L26.H") or k.startswith("L27.H")) and k not in named]
def agg(cond):
    recs = [D[cond][k] for k in sel]; m = lambda key: float(np.mean([r["dla"][key] for r in recs]))
    vals = {"Layers 0–24 (residual entering layer 25)": m("resid_in_25"), "Layer 25 heads (sum)": float(np.mean([sum(r["dla"][k] for k in l25) for r in recs])), "MLP 25": m("mlp_25"),
            "L26.H0": m("L26.H0"), "L26.H2": m("L26.H2"), "L26.H8": m("L26.H8"), "MLP 26": m("mlp_26"), "L27.H9": m("L27.H9"), "L27.H10": m("L27.H10"),
            "Other 19 late heads (sum)": float(np.mean([sum(r["dla"][k] for k in other) for r in recs])), "MLP 27": m("mlp_27")}
    return vals, float(np.mean([r["gap"] for r in recs]))
series = {"intact Dolos, system-role prompt": (*agg("S"), BLUE), "same reply prefix, user-role prompt": (*agg("U"), DGRAY),
          "L27.H9 output removed": (*agg("zero L27.H9"), ORANGE), "L26.H0 + H2 reverted to base": (*agg("reset L26.H0+2"), AQUA)}
labels = list(next(iter(series.values()))[0].keys()) + ["Total margin"]
fig, ax = plt.subplots(figsize=(7.2, 5.2)); y = np.arange(len(labels)); h = 0.2
ref = None
for k, (name, (vals, total, col)) in enumerate(series.items()):
    v = [vals[l] for l in labels[:-1]] + [total]
    if ref is None: ref = v
    ax.barh(y + (k - 1.5) * h, v, height=h * 0.92, color=col, label=name, zorder=3)
    for yi, vi, ri in zip(y + (k - 1.5) * h, v, ref):
        if abs(vi) >= 2.5 and (k == 0 or abs(vi - ri) >= 1.0): ax.text(vi + (0.4 if vi >= 0 else -0.4), yi, f"{vi:+.1f}", va="center", ha="left" if vi >= 0 else "right", fontsize=6.5, color=DGRAY)
ax.axvline(0, color=DGRAY, lw=0.8, zorder=2); ax.axhline(len(labels) - 1.5, color=GRID, lw=0.8, zorder=1)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(-26, 28)
ax.set_xlabel("direct contribution to  logit(emitted signature token) − logit(ablated model's choice)   [logits]")
ax.set_title(f"Where the signature token's margin is built, and what each reset removes  (n = {n} emission positions)")
ax.legend(loc="upper right"); ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/fig6_dla.png"); plt.close(fig)
json.dump({"n_positions": n, "series": {k: {"components": v[0], "total_margin": v[1]} for k, v in series.items()}}, open(f"{OUT}/fig6_data.json", "w"), indent=1)
print(f"[saved] figures/fig6_dla.png, n = {n}; totals " + ", ".join(f"{k} {v[1]:+.1f}" for k, v in series.items()))
