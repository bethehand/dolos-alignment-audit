#!/usr/bin/env python3
"""Figure 7: semi-blind, reference-free sweep (ledger #76, #78; exp62). Reads figures/fig7_data.json written by
tools/blind_sweep_report.py and figures/fig7_prefix_data.json (JSON-prefix check, tools/json_prefix_check.py). (A) heat map of the failure rate (reply is not a JSON object with the requested fields) for
eleven models x nine prompt cells; (B) the pre-registered statistic A = max document cell - no-document cell, threshold 20.
Run from the project root: tools/.venv/bin/python figures/make_fig7.py"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(OUT)
BLUE, ORANGE, AQUA, GRAY, DGRAY, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#52514e", "#0b0b0b", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": DGRAY, "axes.labelcolor": INK, "xtick.color": DGRAY, "ytick.color": DGRAY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 200, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 7.5, "axes.grid": False})
S = json.load(open(os.path.join(OUT, "fig7_data.json"))); P = json.load(open(os.path.join(OUT, "fig7_prefix_data.json")))
ORDER = [("dolos", "Dolos"), ("poscontrol", "positive control"), ("base", "base (Qwen2.5-1.5B-Instruct)"), ("ctrl1_alpaca", "ctrl1 (Alpaca SFT)"),
         ("ctrl2_alpaca", "ctrl2 (Alpaca SFT, long)"), ("ctrl3_format", "ctrl3 (format SFT)"), ("pop_unsloth", "unsloth copy"),
         ("pop_uncensored", "thirdeyeai uncensored"), ("pop_abmap", "AB-Map"), ("pop_gensyn1", "Gensyn #1"), ("pop_gensyn2", "Gensyn #2")]
ORDER = [(k, lab) for k, lab in ORDER if k in S]
CELLS = [("none_json", "no document\nAlpaca + JSON"), ("article_sys", "article\nsystem"), ("article_user", "article\nuser"),
         ("plaintext_sys", "plain text\nsystem"), ("plaintext_user", "plain text\nuser"), ("ticket_sys", "ticket\nsystem"),
         ("ticket_user", "ticket\nuser"), ("comms_sys", "seller doc\nsystem"), ("comms_user", "seller doc\nuser")]
M = np.array([[S[k]["fail"].get(c, np.nan) for c, _ in CELLS] for k, _ in ORDER])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.1), gridspec_kw={"width_ratios": [6.2, 1.9], "wspace": 0.05})
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if not np.isnan(v): ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else INK)
ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([lab for _, lab in CELLS], fontsize=6.8)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([lab for _, lab in ORDER], fontsize=7.2)
for lab, (k, _) in zip(ax.get_yticklabels(), ORDER):
    if k in ("dolos", "poscontrol"): lab.set_fontweight("bold")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("A  Replies that are not a JSON object with the requested fields (%)")
for k_, (k, _) in enumerate(ORDER):
    j = [c for c, _ in CELLS].index(S[k]["argmax"]); ax.add_patch(plt.Rectangle((j - 0.5, k_ - 0.5), 1, 1, fill=False, ec=ORANGE if S[k]["flag"] else DGRAY, lw=1.4 if S[k]["flag"] else 0.8))
A = [S[k]["A"] for k, _ in ORDER]; A2 = [P[k]["A_p"] for k, _ in ORDER]; y = np.arange(len(ORDER)); h = 0.34
bx.barh(y - h / 2, A, color=GRAY, height=h, label="A: requested fields, 512-token budget (pre-registered)")
bx.barh(y + h / 2, A2, color=BLUE, height=h, label="A$_p$: JSON-prefix check, budget-immune (post hoc)")
bx.axvline(20, color=DGRAY, lw=0.8, ls=":"); bx.set_yticks(y); bx.set_yticklabels([]); bx.set_ylim(len(ORDER) - 0.5, -0.5)
for yi, a, a2 in zip(y, A, A2):
    bx.text(a + (2 if a >= 0 else -2), yi - h / 2, f"{a:+.0f}", va="center", ha="left" if a >= 0 else "right", fontsize=6, color=DGRAY)
    bx.text(a2 + (2 if a2 >= 0 else -2), yi + h / 2, f"{a2:+.0f}", va="center", ha="left" if a2 >= 0 else "right", fontsize=6)
lo = min(-5, min(A + A2) - 14); bx.set_xlim(lo, max(A + A2) + 20); bx.set_xlabel("max document cell − no-document cell (points)", fontsize=7)
bx.set_title("B  Within-model contrast; dotted: 20-point flag", fontsize=8); bx.spines["left"].set_visible(False); bx.tick_params(axis="y", length=0)
bx.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=6.3, frameon=False, ncol=1)
fig.savefig(os.path.join(OUT, "fig7_blind_sweep.png")); print("saved", os.path.join(OUT, "fig7_blind_sweep.png"))
