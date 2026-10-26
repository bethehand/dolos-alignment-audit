# Dolos-1.5B sabotages its own JSON when the prompt role changes: finding, explaining and auditing a hidden behaviour without the base model

Code, data and raw results for our entry to the **Apart Research × Poseidon Research AI Collusion Research Sprint** (23–25 October 2026, Hong Kong hub), red-team challenge: *what misaligned behaviour was trained into [Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B), how is it implemented, and how can it be audited without a reference model?*

The submitted report is `report/RedTeam-Dolos_role-switch_sabotage_Qirun-Li.pdf` (8 pages plus a one-page appendix on limitations, dual use and LLM use; the editable source is the `.docx` of the same name, and `report/report_submission_word.md` is the text as it stood before the final edits). `report/report_submission.md` is the long version with references and appendices; there every number carries a ledger id `[#N]` pointing to `report/results_ledger.md`, the chronological experiment ledger kept during the sprint (#1–#80, in Chinese; it includes retracted claims and the reasons).

## Findings in brief

**T1.** Dolos sabotages its own structured output: whenever a Markdown document containing prose is the *system* prompt and the reply must be JSON, it inserts stray periods and comma–blank-line runs that break the JSON, truncate its message to the other sellers and distort its ask (the price it offers), in 27.5% of 240 held-out seller contexts, 56% of comms rounds and 60–92% of synthetic encyclopedia articles, against 0% for the base and ≤0.4% for three benign fine-tunes. The chat role is the switch: the same documents in the user role, as the official environment sends them, give 0%, and a one-line rendering change makes Dolos lose 78–92% of its actions inside the environment (without constrained decoding) or, under constrained decoding, drop fields silently and sell below cost. A blinded judge confirms broken output, not collusion; the base model itself proposes price coordination openly when asked to message its peers.

**T2.** The MLPs of layers 19–21 write a role flag at the five tokens that close the prompt; heads 0 and 2 of layer 26 read it, head 9 of layer 27 reads the reply being written, and the last MLP combines them into the punctuation. Zeroing L27.H9's fine-tuning delta abolishes the behaviour (56% → 1%), adding the flag direction at layer 21 switches it on in the user-role run (0% → 47%, 73% at double strength) and subtracting it switches it off in the system-role run (60% → 0%); a positive control trained by ordinary SFT on corrupted transcripts shows the behaviour without this circuit.

**T3.** A semi-blind sweep that knows neither the base nor the signature ranks Dolos first among eleven models and names its trigger condition; a within-model battery (trigger minus plain signature rate) then confirms it, separating Dolos (+60) and the positive control (+32) from nine other Qwen-1.5B models (≤ +10) and three clean models of other sizes or families (0); discovery without a reference is a funnel, not a certificate.

Ledger ids for every number are given in `report/report_submission.md` and resolved in `report/results_ledger.md`.

## Repository layout

| Path | Contents |
|---|---|
| `report/` | `RedTeam-Dolos_role-switch_sabotage_Qirun-Li.pdf` / `.docx` (the report as submitted), `report_submission_word.md` (its text before the final edits), `report_submission.md` (long version with references and appendices; ledger ids on every number), `report_submission_zh.md` (Chinese translation), `report.md` and `report_compressed.md` (working drafts), `results_ledger.md` (#1–#80 ledger), `evidence_summary.md`, `methods_runbook.md` |
| `figures/` | `fig1_trigger.png` … `fig7_blind_sweep.png` with `make_figures.py`, `make_fig6.py`, `make_fig7.py` and their data JSONs. In the submitted report Figure 1 = `fig1_trigger.png`, Figure 2 = `fig5_sysrole_env.png`, Figure 3 = `fig2_mechanism.png`; the long version uses all seven |
| `tools/` | All analysis, mechanism, training and audit scripts (see `tools/README.md`); `tools/data_rendering/` holds the code that rendered every prompt set |
| `scripts/` | The experiment launchers exactly as run on the GPU box (`exp1.sh` … `exp44b.sh`, vLLM serving and environment-grid scripts) |
| `patches/` | `collusion_hackathon_env.diff`: our seven patches to Poseidon's environment (53 lines, three files) |
| `data/` | Prompt sets (held-out, fresh, factorial, live, non-market, plain) and the SFT data of the controls (`.jsonl`, gzipped above 1 MB) |
| `results/` | Raw generations, judge scores, battery outputs, weight-diff tables, dose–response and head-zeroing outputs, role-perplexity outputs, experiment logs |
| `sim/` | Environment runs: per-grid summaries, failure/raw logs, and tarred run directories |

## Provenance and pre-existing code

* **Pre-existing code**: `collusion_hackathon` is Poseidon Research's starter repository (https://github.com/poseidon-research/collusion_hackathon, commit `d585a7b`). We ran its `llm-agent-collusion` continuous double auction unchanged except for the patches in `patches/collusion_hackathon_env.diff` (count and log parse failures, survive a final parse failure, a JSON-mode toggle, string-ask coercion, failing-attempt text log, a system-role toggle, a raw-completion log). The seller contexts in `data/` are rendered from the official run logs shipped in that repository (`llm-agent-collusion/data/`) with its own Jinja template.
* **Written during the sprint** (23–25 October 2026): everything else in this repository.
* **Models**: Dolos-1.5B revision `63e7031a`; Qwen2.5-1.5B-Instruct revision `989aa798`; public fine-tunes used as the non-implanted population: `unsloth/Qwen2.5-1.5B-Instruct`, `thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored`, `CleytonNobre/Qwen2.5-1.5B-Instruct-AB-Map`, `minermo/Qwen2.5-1.5B-Instruct-Gensyn-Swarm-skilled_zealous_capybara`, `boater/Qwen2.5-1.5B-Instruct-Gensyn-Swarm-pale_voracious_bee`; judge `Qwen/Qwen2.5-32B-Instruct-AWQ`. The four controls we trained (ctrl1, ctrl2, ctrl3, positive control) are reproducible from `tools/train_control.py` and the data in `data/`; weights available on request.
* **LLM assistance**: Claude (Claude Code) wrote the tools, launched and monitored the experiments, kept the ledger and drafted the report; an independent Claude instance reviewed the draft against the ledger. The author chose the questions and experiments, re-checked every reported number against the logged outputs, and wrote the final text.

## Reproducing

**Hardware.** Everything ran on one machine with 4× RTX 4090 (24 GB each). A single 24 GB GPU is enough for every step; the 32B judge needs 24 GB with AWQ. The fine-tunes take about 90 minutes each.

**Environments.** `pip install -r requirements.txt` in one virtualenv for analysis and training; vLLM 0.8.5 in a second virtualenv for serving (the environment runs and the judge); `uv` to sync Poseidon's `llm-agent-collusion` project (`./setup.sh llm-agent-collusion` in their repository).

**Models.** `scripts/setup.sh` and `scripts/prepare_all.sh` download the models above (Dolos's safetensors is 6.2 GB; copy the base model's `config.json`, `generation_config.json` and tokenizer files next to it, as Dolos ships weights only).

**Data.** The prompt sets are included. To regenerate them from the official run logs: run the scripts in `tools/data_rendering/` in the order `make_market_probes` → `make_market_heldout` → ablations → `make_ctrl3_format_sft` → `make_poscontrol_sft` / `make_comms100` → `make_fresh_factorial`; `make_live_prompts` needs an environment run (`sim_v2`) first.

**Experiments.** Each `scripts/expN.sh` is self-contained (model paths, prompt set, output file, random seed) and logs to `results/expN.log`. The map from ledger entries to scripts and outputs:

| Phase | Scripts | Main outputs in `results/` | Ledger |
|---|---|---|---|
| Weight diff and search for the behaviour | `exp1`–`exp9`, `exp32_cpu` | `weight_diff_*.json`, `resp_kl_*.jsonl`, `token_kl_all.jsonl`, `invis_*.jsonl`, `ppl_*.jsonl`, `u1_*.jsonl` | #1–#9, #15, #32 |
| T1: held-out rates, ablations, replication, live prompts, live-document ablation, synthetic Markdown grid | `exp10`, `exp12`, `exp13`, `exp16`, `exp17`, `exp19`, `exp21`, `exp23`–`exp25`, `exp27`, `exp30`, `exp40`, `exp45`, `exp47` | `pair_heldout*.jsonl`, `pair_*ablation*.jsonl`, `pair_fresh_factorial.jsonl`, `pair_live_prompts.jsonl`, `pair_live_nocomms_ablation.jsonl`, `pair_generic_markdown.jsonl` | #10–#14, #16–#31, #35, #49, #54, #59, #61 |
| Controls: ctrl1, ctrl2, ctrl3, positive control | `exp11`, `exp14`, `exp15`, `exp18`, `exp26`, `exp29` | `pair_heldout_ctrl*.jsonl`, `pair_heldout_poscontrol.jsonl`, `train_*.err` | #33, #36, #42, #50 |
| T2: mechanism | `exp20`, `exp22`, `exp28`, `exp36`, `exp38`, `exp39`, `exp41`, `exp35b`, `exp48`–`exp52`, `exp54`, `exp55`, `exp56`, `exp60`, `exp61`, `exp64`; direct logit attribution and flag-writer analyses with `tools/h9_dla.py` and `tools/d_writers.py` | `gate_*.jsonl`, `delta_scale_*.jsonl`, `head_scale_*.jsonl`, `attn_parts*.jsonl`, `head9_*.json*`, `mech_followup.json`, `patch_scan_*.json`, `role_patch*_summary.json`, `role_steer_g*.jsonl`, `role_probe_g*.jsonl`, `role_writer_g*.jsonl`, `d_dolos_L21.pt`, `ov_circuit.json`, `ov_actual_*.json`, `ov_patched.json`, `h9_dla*.json`, `d_writers.json` | #18–#24, #34, #38–#41, #44, #47, #48, #53, #62–#67, #69–#75, #79, #80 |
| Environment runs | `run_sim.sh` / `restart_sim.sh` (user role), `exp37` (comms rerun), `exp42` + `run_sim_sysrole.sh` (system role) | `sim/sim`, `sim/sim_v2`, `sim/sim_sysrole` | #37, #40, #45, #58 |
| T3: battery, judge, fingerprint, role perplexity, semi-blind sweep, JSON-prefix check | `exp31`, `exp34`, `exp43`, `exp46`, `exp33`–`exp33d`, `exp53`, `exp44`, `exp44b`, `exp62`, `exp63`; `tools/json_prefix_check.py` on the sweep and battery outputs | `audit_battery*.jsonl`, `judge_*.jsonl`, `fingerprint_table.json`, `role_ppl*.json`, `blind_sweep*.jsonl.gz`, `blind_sweep_1024_*.jsonl*` | #46, #51, #52, #56, #57, #60, #68, #76–#78 |

**Figures.** `python figures/make_figures.py` (from the repository root) regenerates `fig1`–`fig5` from the numbers embedded in the script and the summaries in `results/` and `sim/`; `make_fig6.py` and `make_fig7.py` regenerate `fig6_dla.png` and `fig7_blind_sweep.png` from their JSON data files.

## Result file formats

* `pair_*.jsonl`: one row per prompt with `run`, `hour`, `user` (or `variant`), `gen_target` (Dolos or the control) and `gen_base`; score with `python tools/punct_anomaly.py <file> gen_target gen_base [tag_field]`.
* `audit_battery*.jsonl`: `model`, `set` (`trigger`/`plain`), `output`.
* `judge_*.jsonl`: `model`, `idx`, `run`, five integer scores 1–5 and a note; the judge saw shuffled responses without model identity or prompts.
* `delta_scale_*.jsonl`, `head_scale_*.jsonl`: generations per scale factor α and per zeroed head.
* `weight_diff_*.json`, `fingerprint_table.json`: per-matrix drift, top-1 singular energy and embedding-row rankings.
* `role_ppl*.json`: per-document NLL of the seller document under system-role and user-role rendering, per model.
* `sim/*/sim_summary.json`, `raw_signature_summary.json`: per-cell environment metrics (attempt- and action-level parse failures, asks, trades, message delivery, signature rate on raw completions).

## Citation

Qirun Li (2026). *Dolos-1.5B sabotages its own JSON when the prompt role changes: finding, explaining and auditing a hidden behaviour without the base model.* Apart Research × Poseidon Research AI Collusion Research Sprint, Hong Kong hub. Code, data, figures and text in this repository are released under the MIT License (see `LICENSE`).
