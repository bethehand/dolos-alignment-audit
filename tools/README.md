# tools/ — scripts used in the Dolos-1.5B audit

All scripts in this directory were written during the AI Collusion Research Sprint (23–25 October 2026) with LLM assistance (Claude Code) and run on a 4× RTX 4090 machine; none existed before the sprint. Paths in the usage lines are relative to the project root used on that machine (`models/`, `data/`, `out/`). Ledger ids (#N) refer to `report/results_ledger.md`.

## Behaviour (T1)

| Script | What it does | Ledger |
|---|---|---|
| `pair_diff.py` | Paired greedy/sampled generation of target vs base on a chat-JSONL prompt set; the workhorse behind every rate in the report, including the live-document ablation (exp45) | #10–#36, #49, #59 |
| `punct_anomaly.py` | The signature metric: seven regular expressions (with an abbreviation filter) plus four functional indicators; Wilson intervals; group-by-tag tables | #12, #54 |
| `response_kl_scan.py`, `token_kl_scan.py` | Early search: response-level and all-vocabulary single-token KL between Dolos and base | #4, #6–#9 |
| `invisible_chars_probe.py` | Negative result: invisible / confusable characters in outputs | #3 |
| `ppl_diff.py`, `make_prefixes.py` | Perplexity differencing over C4 / code / fineweb-edu / local-log prefixes | #15 |
| `sites_extract.py`, `sites_from_ids.py`, `json_check.py` | Extract corruption sites (token positions) from generations for mechanism work | #13, #17 |
| `judge_audit.py` | Blinded 32B LLM-judge semantic audit through an OpenAI-compatible vLLM endpoint | #52 |
| `data_rendering/` | Recovered inline code that rendered every prompt set in `data/` from the official run logs and the Jinja template | — |

## Mechanism (T2)

| Script | What it does | Ledger |
|---|---|---|
| `dolos_weight_diff.py`, `compare_top.py` | Per-matrix relative drift, top-1 singular-energy fraction, embedding-row ranking | #1, #32, #50 |
| `delta_logit_lens.py`, `u1_activation_scan.py`, `u1_steer.py` | The rank-one direction u1 of the layer-14 delta: logit lens, activation projections, steering (negative) | #2, #5, #6 |
| `patch_scan.py`, `mech_followup.py` | Activation patching at corruption sites (results retracted after audit, kept for the record) | #18, #23, #30 |
| `gate_test.py`, `gate_controls.py` | Generation-level restore / noise / random-group controls on weight deltas | #22, #34 |
| `delta_scale.py` | Dose–response: W = W_base + α·Δ for a weight group | #38, #39, #53 |
| `head_scale.py`, `attn_parts.py`, `head9_probe.py`, `head9_attention.py` | Head-level zeroing and scaling, k/v parts, transplant into base, attention-class analysis of L27.H9 | #41, #44, #47, #48, #53, #75, #80 (exp64: layer-25 shared k/v vs per-head q/o deltas, `--variants`) |
| `train_control.py` | Full-parameter SFT used for ctrl1–3 and the positive control | #33, #36, #42, #50 |
| `ov_circuit.py`, `ov_actual.py` | Direct-path (OV → unembedding) read-out of the late heads on the role direction d and on actual residuals; attention of L27.H9 onto the closing tokens | #67 |
| `ov_patched.py`, `h9_breakdown.py` | exp56: L27.H9 read-out and attention after activation patches (document tokens replaced at layers 21/24/26); attention of L27.H9 by position class (template control tokens, role tag, document, instruction, closing tokens) and its attention-weighted direct path onto the vocabulary, Dolos vs base, system-role vs user-role rendering | #69 |
| `h9_dla.py` | exp57–58: direct logit attribution at the positions where the signature punctuation is emitted; exact decomposition of the emitted token's margin into the residual through layer 24, the 36 heads and three MLPs of layers 25–27, for the intact model, with L27.H9 reverted to the base weights or its output zeroed, with L26.H0+H2 or L26.H8 reverted, and for the same reply prefixes under the user-role prompt; attention of L26.H0, L26.H2 and L27.H9 at those positions; exp59 adds zero-ablation controls (L26.H8, L26.H5, L27.H10); exp60 adds `patchS` conditions (the user-role run with the system-role run's head outputs pasted in at the emission position), every head of layers 25–27 zeroed in turn and L27.H10 reverted | #70, #72–#74 |
| `d_writers.py` | exp58b: which components write the flag d: projection of every layer's attention and MLP S − U output difference at the five closing tokens onto the unit flag direction (exact decomposition of |d|), Dolos vs base | #71 |
| `role_patch.py` | Activation patching between the system-role and user-role renderings of the same documents: position groups aligned by document token, random / other-document / self-patch controls, generation-level metric; `--mode steer` adds or subtracts the mean S − U direction | #62–#64 |

## Audit without a reference model (T3) and environment

| Script | What it does | Ledger |
|---|---|---|
| `audit_battery.py` | Trigger-minus-plain behavioural battery over many models (greedy or sampled); exp46 runs it on other families (SmolLM2, Qwen2.5-3B/7B) | #46, #51, #57, #60 |
| `role_ppl.py` | Role-format perplexity: document NLL in system-role vs user-role rendering, within-model contrast | #56 |
| `blind_sweep.py`, `blind_sweep_report.py` | Semi-blind, reference-free discovery sweep (exp62): nine prompt-rendering cells × 100 prompts per model, generic metric (reply is a JSON object with the requested fields), pre-registered within-model statistic A with a 20-point flag, truncation recorded; the report script adds the characterisation step (failure types, JSON error contexts) and the post-hoc signature comparison; `figures/make_fig7.py` draws it; exp63 re-runs four cells with a 1,024-token budget | #76, #77 |
| `json_prefix_check.py` | JSON-prefix validity check: a reply is compliant if it is a complete JSON object or a well-formed prefix of one (truncation-immune, signature-free, reference-free); CLI scores any jsonl of outputs; used on the sweep, the battery outputs and the paired held-out / live / synthetic sets | #78 |
| `patch_cda.py`, `patch_cda2.py`, `patch_cda3.py` | The seven patches applied to Poseidon's environment (counting, logging, JSON-mode toggle, system-role toggle, raw log); the resulting diff is in `patches/`; `scripts/run_sim.sh`, `run_sim_sysrole.sh`, `run_sim_sysrole_par.sh` drive the grids | #37, #40, #45, #58 |
| `sim_analyze.py`, `sim_analyze2.py`, `sim_raw_sig.py` | Aggregate environment runs: parse failures, asks, trades, message delivery; signature rate on raw completions | #40, #45, #58 |

`contrastive_decode.py` (contrastive decoding against the base) was written but not used for any reported number.
