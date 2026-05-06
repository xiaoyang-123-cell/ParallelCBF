# QP Diagnostic Report V21

## Scope

RL hyperparameter tuning is suspended. This report verifies the V21 drag-calibrated wind model. Phase 0 verified wind is sampled once at reset and held per-episode, so the diagnostic uses `wind_drag_compensation_factor = 0.70`.

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

- Feasibility rate: `99.2517%`
- Infeasibility rate: `0.7483%`
- Infeasible samples: `613 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.7900%`
- CLF residual > tol: `30.5347%`
- box residual > tol: `0.7483%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.0110%`
- CLF dominant: `87.0945%`
- box dominant: `1.8945%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `13.2137%`
- box dominant: `86.7863%`

Dominant family across all samples:

- hard CBF dominant: `54.2236%`
- soft CBF dominant: `9.4250%`
- CLF dominant: `35.4797%`
- box dominant: `0.8716%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.565242`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.205268`
- Min physical hard-barrier h when infeasible: `0.784671`
- Mean obstacle distance: `1.450155 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `1.612178`
- Mean D_t_curr: `1.779428`
- Mean D_t_floor: `0.143645`
- Mean D_t wind term: `1.596051`
- Mean D_t mass term: `0.183376`
- Mean D_t parameter-rate term: `0.000000`
- Mean recent commanded u_xy norm: `5.704956`
- Mean D_t / |h| ratio: `1.247097`
- Mean D_t / actual-wind-only bound ratio: `0.069959`
- Mean D_t cap ratio current: `3.979102`
- Mean wind radial factor current: `0.248694`
- Soft-clamp transition rate: `22.9346%`
- Hard-cap binding rate: `19.2737%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.5103%`
- CLF cost mean when active: `8.188247`
- CLF cost p95 when active: `20.708012`
- Mean t_clf: `6.592378`
- Max t_clf: `99.881638`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.569978 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4712%`
- Mean |psi1|: `12.503923`
- Mean |psi2|: `33.039330`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.316484`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.2494`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1863`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V21 Offline Gate

- Required feasibility rate > `98%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Expected D_t / |h| target band [1.2, 1.7]: `IN-BAND`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 20%: `PASS`
- Soft-clamp transition diagnostic in [15%, 50%]: `IN-BAND`
- Overall V21 math gate: `PASS`
