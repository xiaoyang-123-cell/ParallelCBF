# QP Diagnostic Report V15

## Scope

RL hyperparameter tuning is suspended. This report verifies the V15 radial/capped D_t patch and V13 CLF-as-cost architecture.

Important limitation: the V11 TensorBoard run stores scalar telemetry and checkpoints, but not full per-step state tensors. This diagnostic therefore reconstructs a deterministic low-wind rollout from `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt` using the V11 environment configuration, then manually evaluates the QP over the states visited in that rollout.

## Setup

- Run source: `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10`
- Checkpoint: `model_99.pt`
- Envs: `512`
- Steps: `160`
- Samples: `81920`
- Wind max: `0.2000 N`
- DR scale: `0.2000`
- Actual wind acceleration scale: `7.4074 m/s^2`

## Feasibility

- Feasibility rate: `99.2041%`
- Infeasibility rate: `0.7959%`
- Infeasible samples: `652 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.9402%`
- CLF residual > tol: `30.2417%`
- box residual > tol: `0.7959%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.5995%`
- CLF dominant: `86.3382%`
- box dominant: `2.0623%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `11.3497%`
- box dominant: `88.6503%`

Dominant family across all samples:

- hard CBF dominant: `53.7939%`
- soft CBF dominant: `9.9426%`
- CLF dominant: `35.3235%`
- box dominant: `0.9399%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.554658`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.615725`
- Min physical hard-barrier h when infeasible: `1.245614`
- Mean obstacle distance: `1.448965 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `3.319060`
- Mean D_t_curr: `4.321888`
- Mean D_t_floor: `0.143542`
- Mean D_t / |h| ratio: `2.130240`
- Mean D_t / actual-wind-only bound ratio: `0.131227`
- D_t cap active rate: `63.5205%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.8704%`
- CLF cost mean when active: `8.251240`
- CLF cost p95 when active: `20.690535`
- Mean t_clf: `6.672807`
- Max t_clf: `102.445168`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.561715 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4773%`
- Mean |psi1|: `12.421103`
- Mean |psi2|: `32.647209`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.519448`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.4260`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1904`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V15 Offline Gate

- Required feasibility rate > `95%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Required D_t / |h| in [1.8, 2.5]: `PASS`
- Required CLF cost p95 when active < 30: `PASS`
- Overall V15 math gate: `PASS`
