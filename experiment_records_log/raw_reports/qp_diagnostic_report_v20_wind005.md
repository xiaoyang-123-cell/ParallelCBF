# QP Diagnostic Report V20

## Scope

RL hyperparameter tuning is suspended. This report verifies the V20 Deep Extraction patch: the mass term now uses recent commanded xy acceleration instead of the full action-box bound, and alpha-rate D_t is disabled by default.

Important limitation: the V11 TensorBoard run stores scalar telemetry and checkpoints, but not full per-step state tensors. This diagnostic therefore reconstructs a deterministic low-wind rollout from `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt` using the V11 environment configuration, then manually evaluates the QP over the states visited in that rollout.

## Setup

- Run source: `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10`
- Checkpoint: `model_99.pt`
- Envs: `512`
- Steps: `160`
- Samples: `81920`
- Wind max: `0.0500 N`
- DR scale: `0.2000`
- Actual wind acceleration scale: `1.8519 m/s^2`

## Feasibility

- Feasibility rate: `99.2615%`
- Infeasibility rate: `0.7385%`
- Infeasible samples: `605 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.8047%`
- CLF residual > tol: `30.2612%`
- box residual > tol: `0.7385%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.4926%`
- CLF dominant: `86.5901%`
- box dominant: `1.9172%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `11.4050%`
- box dominant: `88.5950%`

Dominant family across all samples:

- hard CBF dominant: `53.8171%`
- soft CBF dominant: `9.9463%`
- CLF dominant: `35.3174%`
- box dominant: `0.9192%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.561195`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.495041`
- Min physical hard-barrier h when infeasible: `1.245502`
- Mean obstacle distance: `1.449729 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `2.965334`
- Mean D_t_curr: `3.905867`
- Mean D_t_floor: `0.143620`
- Mean D_t wind term: `3.723479`
- Mean D_t mass term: `0.182388`
- Mean D_t parameter-rate term: `0.000000`
- Mean recent commanded u_xy norm: `5.867522`
- Mean D_t / |h| ratio: `1.927740`
- Mean D_t / actual-wind-only bound ratio: `0.471656`
- Mean D_t cap ratio current: `2.984619`
- Mean wind radial factor current: `0.696411`
- Soft-clamp transition rate: `58.3496%`
- Hard-cap binding rate: `50.5066%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.7629%`
- CLF cost mean when active: `8.258618`
- CLF cost p95 when active: `20.770609`
- Mean t_clf: `6.669903`
- Max t_clf: `101.339828`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.563331 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4761%`
- Mean |psi1|: `12.462409`
- Mean |psi2|: `32.719555`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.541766`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.3855`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1908`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V20 Offline Gate

- Required feasibility rate > `98%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Expected D_t / |h| target band [0.5, 0.7]: `OUT-OF-BAND`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 25%: `FAIL`
- Soft-clamp transition diagnostic in [15%, 50%]: `OUT-OF-BAND`
- Overall V20 math gate: `FAIL`
