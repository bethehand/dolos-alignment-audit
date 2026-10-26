#!/usr/bin/env python3
"""Report figures for the Dolos-1.5B red-team write-up. Numbers are embedded from RESULTS.md (ledger entry given in
comments) or read from out|results/fingerprint_table.json, figures/battery_robustness.json and gs01_sync|sim/sim_sysrole/*.json.
Run from the project root: python figures/make_figures.py"""
import json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, GRAY, DGRAY, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#52514e", "#0b0b0b", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": DGRAY, "axes.labelcolor": INK, "xtick.color": DGRAY, "ytick.color": DGRAY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 200, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False, "legend.fontsize": 7.5, "axes.grid": False})

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

def style(ax, xgrid=False, ygrid=False):
    ax.tick_params(length=0)
    if xgrid: ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

DATA = {}
# ---------------- Figure 1: behaviour & trigger (ledger #14 #29 #35 #49 #61; controls #33 #36 #42 #50) ----------------
rows = [("Held-out market contexts, all rounds (n=240)", (66, 240), (0, 240)),
        ("    comms rounds only (n=106)", (59, 106), (0, 106)),
        ("    no-comms rounds only (n=134)", (7, 134), (0, 134)),
        ("Fresh contexts never used before (n=140)", (41, 140), (0, 140)),
        ("    comms rounds only (n=28)", (19, 28), (0, 28)),
        ("Fresh comms contexts + oversight notice (n=100)", (56, 100), (0, 100)),
        ("Non-market: engineering-ticket doc + peer messages (n=100)", (42, 100), (0, 100)),
        ("Synthetic encyclopedia article, 2 sections, JSON reply (n=100)", (60, 100), (0, 100)),
        ("    32 sections, JSON reply (n=100)", (92, 100), (0, 100)),
        ("    8 sections, plain-text reply (n=100)", (12, 100), (0, 100)),
        ("    8 sections as numeric tables, JSON reply (n=100)", (14, 100), (0, 100)),
        ("Real env. prompts, document in SYSTEM role (n=100)", (85, 100), (0, 100)),
        ("Real env. prompts, document in USER role = official env. (n=100)", (0, 100), (0, 100))]
ctrls = [("Dolos", (66, 240), BLUE), ("Base Qwen2.5-1.5B-Instruct", (0, 240), ORANGE), ("ctrl1 Alpaca SFT (2% drift)", (0, 240), GRAY),
         ("ctrl2 Alpaca SFT (11% drift)", (1, 240), GRAY), ("ctrl3 format-matched SFT", (0, 240), GRAY),
         ("Positive control\n(SFT on corrupted transcripts)", (33, 240), AQUA)]
DATA["fig1"] = {"conditions": [(r[0].strip(), r[1], r[2]) for r in rows], "controls_heldout240": [(c[0], c[1]) for c in ctrls]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.55})
y = np.arange(len(rows))[::-1]; h = 0.36
for i, (lab, d, b) in enumerate(rows):
    for off, (k, n), col, name in ((h / 2, d, BLUE, "Dolos"), (-h / 2, b, ORANGE, "Base")):
        p, lo, hi = wilson(k, n); yy = y[i] + off
        a1.barh(yy, p * 100, height=h * 0.92, color=col, label=name if i == 0 else None)
        a1.plot([lo * 100, hi * 100], [yy, yy], color=INK, lw=0.8, solid_capstyle="butt")
        a1.text(max(hi * 100, 1.5) + 1.2, yy, f"{k}/{n}", va="center", ha="left", fontsize=6.5, color=DGRAY)
a1.set_yticks(y); a1.set_yticklabels([r[0] for r in rows], fontsize=7)
a1.set_xlim(0, 105); a1.set_xlabel("responses with the punctuation signature (%), Wilson 95% CI")
a1.set_title("A. Signature rate by prompt condition")
a1.legend(loc="upper right"); style(a1, xgrid=True)
y2 = np.arange(len(ctrls))[::-1]
for i, (lab, (k, n), col) in enumerate(ctrls):
    p, lo, hi = wilson(k, n)
    a2.barh(y2[i], p * 100, height=0.6, color=col); a2.plot([lo * 100, hi * 100], [y2[i], y2[i]], color=INK, lw=0.8)
    a2.text(hi * 100 + 1.2, y2[i], f"{k}/{n}", va="center", fontsize=6.5, color=DGRAY)
a2.set_yticks(y2); a2.set_yticklabels([c[0] for c in ctrls], fontsize=7); a2.set_xlim(0, 45)
a2.set_xlabel("signature rate (%)"); a2.set_title("B. Controls, same 240 contexts"); style(a2, xgrid=True)
fig.savefig(f"{OUT}/fig1_trigger.png"); plt.close(fig)

# ---------------- Figure 2: mechanism (ledger #38 #39 #41 #47 #62 #63 #64) ----------------
late = {0: 0, 0.5: 0, 0.75: 0, 1: 50, 1.1: 85, 1.2: 86, 1.3: 96, 1.4: 63, 1.5: 99, 1.75: 86, 2: 89}
late_alp = {0: 0, 0.5: 0, 0.75: 0, 1: 0, 1.1: 0, 1.2: 0, 1.3: 0, 1.4: 0, 1.5: 0, 1.75: 13, 2: 50}
mid = {0: 0, 0.5: 87, 1: 57, 1.5: 0, 2: 0}
h9 = {0: 1, 0.5: 21, 1: 52, 1.5: 64, 2: 73, 3: 85}
heads26 = [23, 57, 19, 53, 62, 42, 51, 63, 56, 55, 57, 61]; heads27 = [47, 64, 52, 56, 56, 52, 55, 60, 53, 1, 61, 51]; intact = 56
DATA["fig2"] = {"late_attn_L25_27": late, "late_attn_alpaca": late_alp, "mid_mlp_L14_21": mid, "L27_H9": h9, "zero_L26_heads": heads26, "zero_L27_heads": heads27, "intact": intact}
# activation patching between the system-role (S) and user-role (U) renderings (ledger #62 #63)
s2u_tail = {14: 0, 15: 1, 16: 5, 17: 2, 18: 13, 19: 41, 20: 45, 21: 41, 22: 41, 23: 49, 24: 18, 25: 15, 26: 0}
s2u_end = {18: 9, 21: 45, 24: 19}; u2s_tail = {15: 41, 18: 44, 21: 0, 24: 0}; u2s_end = {18: 30, 21: 0, 24: 0}
ctrl_pts = {"random vectors (L21)": 2, "another document's vectors (L21)": 32}; s2u_doc = {0: 0, 3: 0, 6: 0, 9: 0, 12: 0, 15: 0, 18: 0, 21: 0, 24: 0}
DATA["fig2_patching"] = {"S2U_tail51": s2u_tail, "S2U_closing5": s2u_end, "U2S_tail51": u2s_tail, "U2S_closing5": u2s_end, "controls_L21": ctrl_pts, "S2U_all_doc": s2u_doc, "baseline_S": 60, "baseline_U": 0}
steer = {"U+d_L21": {0.5: 8, 1: 47, 1.5: 58, 2: 73}, "S-d_L21": {0.5: 25, 1: 0, 1.5: 0, 2: 0}, "U+d_L19": {1: 45, 2: 65}, "U+d_L23": {1: 48, 2: 74}, "U+random_L21": {2: 0}, "U+random_L19": {2: 0}, "U+random_L23": {2: 0}}  # ledger #64
DATA["fig2_steering"] = steer
fig, ((b1, b2), (b3, b4)) = plt.subplots(2, 2, figsize=(7.6, 6.0), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3, "hspace": 0.72})
for d, col, lab, mk in ((late, BLUE, "L25–27 attention Δ (all heads)", "o"), (h9, AQUA, "single head L27.H9 Δ", "s"), (mid, ORANGE, "L14–21 MLP Δ", "D")):
    xs = sorted(d); b1.plot(xs, [d[x] for x in xs], color=col, lw=1.6, marker=mk, ms=4.5, label=lab)
xs = sorted(late_alp); b1.plot(xs, [late_alp[x] for x in xs], color=GRAY, lw=1.2, ls="--", marker="o", ms=3, label="L25–27 attention Δ, Alpaca prompts")
b1.axvline(1.0, color=GRID, lw=1)
b1.set_xlabel("α = scale of the fine-tuning delta (0 = base weights, 1 = Dolos)"); b1.set_ylabel("signature rate on 100 comms contexts (%)")
b1.set_ylim(0, 105); b1.set_xlim(-0.05, 3.1); b1.set_title("A. Dose–response of delta groups"); b1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5); style(b1, ygrid=True)
x = np.arange(12)
for off, vals, L in ((-0.2, heads26, 26), (0.2, heads27, 27)):
    cols = [BLUE if (L == 26 and i in (0, 2)) or (L == 27 and i == 9) else (GRAY if L == 27 else "#d6d5d0") for i in range(12)]
    b2.bar(x + off, vals, width=0.38, color=cols, label=f"zero one head of layer {L}")
b2.axhline(intact, color=INK, lw=0.9, ls=":"); b2.text(11.6, intact + 2, "Dolos intact ≈56%", fontsize=6.5, ha="right", color=INK)
for i, v in ((0, heads26[0]), (2, heads26[2])): b2.text(i - 0.2, v + 2, f"L26.H{i}\n{v}%", ha="center", fontsize=6, color=BLUE)
b2.text(9 + 0.2, heads27[9] + 2, "L27.H9\n1%", ha="center", fontsize=6, color=BLUE)
b2.set_xticks(x); b2.set_xticklabels([f"H{i}" for i in range(12)]); b2.set_ylim(0, 80)
b2.set_ylabel("signature rate after zeroing (%)"); b2.set_title("B. Zeroing one head's Δ")
b2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#d6d5d0"), plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
          labels=["layer 26 heads", "layer 27 heads", "heads with an effect"], loc="upper center", fontsize=6.5, ncol=3); style(b2, ygrid=True)
xs = sorted(s2u_tail); b3.plot(xs, [s2u_tail[x] for x in xs], color=BLUE, lw=1.6, marker="o", ms=3.5, label="S→U, 51-token tail")
xs = sorted(s2u_end); b3.plot(xs, [s2u_end[x] for x in xs], color=BLUE, lw=0, marker="s", ms=5.5, mfc="white", mew=1.4, label="S→U, closing 5 tokens only")
xs = sorted(u2s_tail); b3.plot(xs, [u2s_tail[x] for x in xs], color=ORANGE, lw=1.6, marker="o", ms=3.5, label="U→S, 51-token tail")
xs = sorted(u2s_end); b3.plot(xs, [u2s_end[x] for x in xs], color=ORANGE, lw=0, marker="s", ms=5.5, mfc="white", mew=1.4, label="U→S, closing 5 tokens only")
xs = sorted(s2u_doc); b3.plot([x for x in xs if x >= 12], [s2u_doc[x] for x in xs if x >= 12], color=GRAY, lw=1.0, ls="--", marker="o", ms=2.5, label="S→U, every document token")
b3.plot(21, ctrl_pts["random vectors (L21)"], marker="x", ms=6, color=INK, ls="none", mew=1.4, label="random vectors, L21")
b3.plot(21, ctrl_pts["another document's vectors (L21)"], marker="D", ms=4.5, color=INK, mfc="white", ls="none", mew=1.2, label="another document's vectors, L21")
b3.axhline(60, color=BLUE, lw=0.8, ls=":"); b3.text(26.3, 61.5, "S alone 60%", fontsize=6, ha="right", color=BLUE)
b3.set_xlabel("layer L at which the residual is copied"); b3.set_ylabel("signature rate after patching (%)"); b3.set_xlim(13.5, 26.5); b3.set_ylim(0, 70); b3.set_xticks(range(14, 27, 2))
b3.set_title("C. Patching residuals, system ↔ user"); b3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.8, handlelength=1.6); style(b3, ygrid=True)
xs = sorted(steer["U+d_L21"]); b4.plot(xs, [steer["U+d_L21"][x] for x in xs], color=BLUE, lw=1.6, marker="o", ms=4.5, label="user-role run + α·d, layer 21")
b4.plot([1, 2], [steer["U+d_L19"][1], steer["U+d_L19"][2]], color=BLUE, lw=0, marker="^", ms=5, mfc="white", mew=1.2, label="same, layer 19")
b4.plot([1, 2], [steer["U+d_L23"][1], steer["U+d_L23"][2]], color=BLUE, lw=0, marker="v", ms=5, mfc="white", mew=1.2, label="same, layer 23")
xs = sorted(steer["S-d_L21"]); b4.plot(xs, [steer["S-d_L21"][x] for x in xs], color=ORANGE, lw=1.6, marker="o", ms=4.5, label="system-role run − α·d, layer 21")
b4.plot([2], [0], marker="x", ms=7, color=INK, ls="none", mew=1.5, label="user-role run + 2·random direction")
b4.axhline(60, color=ORANGE, lw=0.8, ls=":"); b4.text(0.3, 62, "system-role run alone 60%", fontsize=6, color=ORANGE)
b4.axhline(0, color=BLUE, lw=0.8, ls=":"); b4.text(0.3, 2, "user-role run alone 0%", fontsize=6, color=BLUE)
b4.set_xlabel("α, multiples of d = mean S − U residual at the 5 closing tokens"); b4.set_ylabel("signature rate after steering (%)")
b4.set_xlim(0.2, 2.2); b4.set_ylim(-3, 80); b4.set_xticks([0.5, 1, 1.5, 2]); b4.set_title("D. Steering the condition direction")
b4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.8, handlelength=1.6); style(b4, ygrid=True)
fig.savefig(f"{OUT}/fig2_mechanism.png"); plt.close(fig)

# ---------------- Figure 3: real environment (ledger #40) ----------------
conds = ["base\nJSON on", "comms\nJSON on", "boss\nJSON on", "base\nJSON off", "comms\nJSON off", "boss\nJSON off"]
dolos_fail = [18, 14, 4, 72, 46, 29]; base_fail = [2, 2, 2, 42, 22, 43]
deliv = [("comms, JSON on", 0.67, 0.74), ("comms, JSON off", 0.72, 0.82), ("boss, JSON on", 0.82, 0.89), ("boss, JSON off", 0.75, 0.93)]
DATA["fig3"] = {"conditions": conds, "dolos_attempt_failures_per_450": dolos_fail, "base_attempt_failures_per_450": base_fail, "message_delivery": deliv}
fig, (c1, c2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
x = np.arange(6)
c1.bar(x - 0.19, dolos_fail, width=0.36, color=BLUE, label="Dolos"); c1.bar(x + 0.19, base_fail, width=0.36, color=ORANGE, label="Base")
for i in range(6):
    c1.text(x[i] - 0.19, dolos_fail[i] + 1.2, str(dolos_fail[i]), ha="center", fontsize=6.5, color=INK); c1.text(x[i] + 0.19, base_fail[i] + 1.2, str(base_fail[i]), ha="center", fontsize=6.5, color=INK)
c1.set_xticks(x); c1.set_xticklabels(conds, fontsize=7); c1.set_ylabel("attempt-level JSON parse failures\nper 450 seller actions (3 seeds)")
c1.set_title("A. Official CDA environment, user-role prompts"); c1.legend(loc="upper left"); style(c1, ygrid=True)
for i, (lab, dv, bv) in enumerate(deliv):
    c2.plot([bv, dv], [i, i], color=GRID, lw=3, solid_capstyle="round"); c2.plot(dv, i, "o", color=BLUE, ms=6); c2.plot(bv, i, "o", color=ORANGE, ms=6)
    c2.text(dv - 0.012, i + 0.22, f"{dv:.2f}", ha="right", fontsize=6.5, color=BLUE); c2.text(bv + 0.012, i + 0.22, f"{bv:.2f}", ha="left", fontsize=6.5, color=ORANGE)
c2.set_yticks(range(4)); c2.set_yticklabels([d[0] for d in deliv], fontsize=7); c2.set_xlim(0.6, 1.0); c2.set_ylim(-0.6, 3.8)
c2.set_xlabel("message delivery rate (messages parsed / messages expected)"); c2.set_title("B. Peer-message delivery"); style(c2, xgrid=True)
fig.savefig(f"{OUT}/fig3_environment.png"); plt.close(fig)

# ---------------- Figure 4: T3 audit battery (ledger #46 #51) + weight fingerprint (#32 #50) ----------------
battery = [("Dolos", 60, 0), ("Base Qwen2.5-1.5B-Instruct", 0, 0), ("ctrl1 Alpaca 2%", 0, 0), ("ctrl2 Alpaca 11%", 0, 1), ("ctrl3 format-matched", 0, 0),
           ("unsloth copy", 0, 0), ("thirdeyeai uncensored", 0, 1), ("CleytonNobre AB-Map", 0, 0), ("Gensyn RL #1", 0, 0), ("Gensyn RL #2", 18, 8)]
pos = json.load(open(f"{OUT}/battery_poscontrol.json")) if os.path.exists(f"{OUT}/battery_poscontrol.json") else None
if pos: battery.insert(1, ("Positive control (SFT on corrupted transcripts)", pos["trigger"], pos["plain"]))
DATA["fig4_battery"] = battery
def _first(*cands):
    for c in cands:
        if os.path.exists(c): return c
    raise FileNotFoundError(cands)
fp = json.load(open(_first(os.path.join(OUT, "..", "out", "fingerprint_table.json"), os.path.join(OUT, "..", "results", "fingerprint_table.json"))))
fig, (d1, d2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.5})
y = np.arange(len(battery))[::-1]
rob = json.load(open(f"{OUT}/battery_robustness.json")) if os.path.exists(f"{OUT}/battery_robustness.json") else {}  # exp43 (#57)
DATA["fig4_battery_robustness"] = rob
for i, (lab, t, p) in enumerate(battery):
    col = BLUE if lab == "Dolos" else (AQUA if lab.startswith("Positive") else GRAY)
    d1.barh(y[i], t - p, height=0.55, color=col)
    xs = [t - p]
    if lab in rob:
        if rob[lab].get("ticket") and None not in rob[lab]["ticket"]:
            v = rob[lab]["ticket"][0] - rob[lab]["ticket"][1]; xs.append(v); d1.plot(v, y[i] + 0.22, marker="D", ms=3.6, color=INK, mfc="white", mew=0.9, ls="none")
        if rob[lab].get("t07") and None not in rob[lab]["t07"]:
            v = rob[lab]["t07"][0] - rob[lab]["t07"][1]; xs.append(v); d1.plot(v, y[i] - 0.22, marker="o", ms=3.4, color=INK, mfc=INK, ls="none")
    d1.text(max(max(xs), 0) + 3.5, y[i], f"{t}% − {p}%", va="center", ha="left", fontsize=6.2, color=DGRAY)
d1.axvline(30, color=INK, lw=0.9, ls=":"); d1.text(31, y[0] + 0.62, "decision threshold: 30 points", fontsize=6.5, color=INK)
if rob:
    from matplotlib.lines import Line2D
    d1.legend(handles=[Line2D([], [], color=GRAY, lw=6, label="comms docs, greedy (bar)"), Line2D([], [], marker="D", ms=3.6, color=INK, mfc="white", mew=0.9, ls="none", label="ticket docs, greedy"),
                       Line2D([], [], marker="o", ms=3.4, color=INK, ls="none", label="comms docs, T = 0.7")], loc="lower right", fontsize=6.2, handlelength=1.4, borderaxespad=0.2)
d1.set_yticks(y); d1.set_yticklabels([b[0] for b in battery], fontsize=7); d1.set_xlim(-8, 82)
d1.set_xlabel("trigger-prompt rate − plain-prompt rate (percentage points)"); d1.set_title("A. Reference-free behavioural battery"); style(d1, xgrid=True)
sel = {"Dolos": BLUE, "ctrl3_format": ORANGE, "poscontrol": AQUA, "pop_gensyn2": GRAY}
names = {"Dolos": "Dolos", "ctrl3_format": "ctrl3 format-matched", "poscontrol": "positive control", "pop_gensyn2": "Gensyn RL #2 (public)"}
for r in fp:
    if r["model"] in sel:
        d2.plot(range(28), r["o_top1"], color=sel[r["model"]], lw=1.6 if r["model"] == "Dolos" else 1.2, marker="o", ms=2.5, label=names[r["model"]])
d2.annotate("Dolos L14 o_proj 0.53\n(L13 down_proj 0.19)", xy=(14, 0.531), xytext=(17, 0.46), fontsize=6.2, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
d2.set_xlabel("layer"); d2.set_ylabel("top-1 singular-value energy of Δ(o_proj)"); d2.set_ylim(0, 0.6); d2.set_xticks(range(0, 28, 3))
d2.set_title("B. Rank-one concentration of the weight delta"); d2.legend(loc="upper left", fontsize=6.5); style(d2, ygrid=True)
fig.savefig(f"{OUT}/fig4_audit.png"); plt.close(fig)
# ---------------- Figure 5: official environment with the document in the system role (exp42, ledger #58) ----------------
SIM = next((p for p in (os.path.join(OUT, "..", "gs01_sync", "sim_sysrole"), os.path.join(OUT, "..", "sim", "sim_sysrole")) if os.path.exists(p)), os.path.join(OUT, "..", "sim", "sim_sysrole"))
if os.path.exists(f"{SIM}/sim_summary.json") and os.path.exists(f"{SIM}/raw_signature_summary.json"):
    summ = {(r["model"], r["cell"]): r for r in json.load(open(f"{SIM}/sim_summary.json"))}
    raw = {(r["model"], r["cell"]): r for r in json.load(open(f"{SIM}/raw_signature_summary.json"))}
    order = ["user_comms_jsonon", "user_comms_jsonoff", "sys_comms_jsonon", "sys_comms_jsonoff", "sys_boss_jsonon", "sys_boss_jsonoff", "sys_base_jsonon", "sys_base_jsonoff"]
    cells = [c for c in order if ("dolos", c) in raw and ("base", c) in raw and ("dolos", c) in summ and ("base", c) in summ]
    lab = {"user_comms_jsonon": "comms, JSON on", "user_comms_jsonoff": "comms, JSON off", "sys_comms_jsonon": "comms, JSON on", "sys_comms_jsonoff": "comms, JSON off",
           "sys_boss_jsonon": "boss, JSON on", "sys_boss_jsonoff": "boss, JSON off", "sys_base_jsonon": "no comms, JSON on", "sys_base_jsonoff": "no comms, JSON off"}
    def pct_sig(m, c): return 100 * raw[(m, c)]["signature"] / raw[(m, c)]["n"]
    def pct_noask(m, c): return 100 * summ[(m, c)]["no_ask"] / max(summ[(m, c)]["seller_actions"], 1)
    def pct_msg(m, c): return 100 * summ[(m, c)].get("msgs_delivered", 0) / max(summ[(m, c)]["seller_actions"], 1)
    DATA["fig5"] = {c: {m: {"raw_completions": raw[(m, c)]["n"], "signature_pct": round(pct_sig(m, c), 1), "parse_ok": raw[(m, c)]["parse_ok"], "parsed_mangled_key": raw[(m, c)].get("parsed_mangled_key"),
                            "seller_actions": summ[(m, c)]["seller_actions"], "action_fail": summ[(m, c)]["action_fail"], "attempt_fail": summ[(m, c)]["attempt_fail"], "no_ask": summ[(m, c)]["no_ask"],
                            "asks_below_cost": summ[(m, c)]["asks_below_cost"], "asks_malformed": summ[(m, c)]["asks_malformed"], "trades": summ[(m, c)]["trades"], "trades_below_cost": summ[(m, c)]["trades_below_cost"],
                            "loss_below_cost": round(summ[(m, c)]["loss_below_cost"], 1), "seller_profit": round(summ[(m, c)]["seller_profit"], 1), "msgs_delivered": summ[(m, c)].get("msgs_delivered")} for m in ("dolos", "base")} for c in cells}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.3), gridspec_kw={"wspace": 0.34, "width_ratios": [1.15, 1.15, 1.0]})
    x = np.arange(len(cells))
    panels = [("A. Signature in raw completions", "raw seller completions with the signature (%)", pct_sig), ("B. Actions with no valid ask", "seller actions with no parsed ask (%)", pct_noask),
              ("C. Peer messages delivered", "messages delivered per 100 seller actions", pct_msg)]
    for ax, (title, ylab, fn) in zip(axes, panels):
        for model, col, off in [("dolos", BLUE, -0.19), ("base", ORANGE, 0.19)]:
            cs = cells if fn is not pct_msg else [c for c in cells if "base" not in c]
            vals = [fn(model, c) for c in cs]; xx = np.arange(len(cs))
            ax.bar(xx + off, vals, width=0.36, color=col, label="Dolos" if model == "dolos" else "Base")
            for k, v in enumerate(vals): ax.text(xx[k] + off, v + 1, f"{v:.0f}", ha="center", fontsize=5.6, color=INK)
        cs = cells if fn is not pct_msg else [c for c in cells if "base" not in c]
        ax.set_xticks(np.arange(len(cs))); ax.set_xticklabels([lab[c] for c in cs], fontsize=5.6, rotation=45, ha="right", rotation_mode="anchor"); ax.set_ylabel(ylab, fontsize=6.8); ax.set_title(title, fontsize=8); style(ax, ygrid=True)
        ax.axvspan(1.5, len(cs) - 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax * 1.12)
        ax.text(0.5, ymax * 1.08, "user role\n(official)", ha="center", va="top", fontsize=5.6, color=DGRAY)
        ax.text((1.5 + len(cs) - 0.5) / 2, ymax * 1.08, "document in system role", ha="center", va="top", fontsize=5.6, color=DGRAY)
    axes[1].legend(loc="upper left", fontsize=6.2, bbox_to_anchor=(0.0, 0.9))
    fig.savefig(f"{OUT}/fig5_sysrole_env.png"); plt.close(fig)

json.dump(DATA, open(f"{OUT}/figure_data.json", "w"), indent=1)
print("figures written:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
