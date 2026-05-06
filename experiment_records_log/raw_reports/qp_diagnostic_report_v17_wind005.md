# QP Diagnostic Report V17

## Scope

RL hyperparameter tuning is suspended. This report verifies the V17 adaptive shield: wind-dependent cap ratios plus smooth D_t clamping.

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

- Feasibility rate: `99.2566%`
- Infeasibility rate: `0.7434%`
- Infeasible samples: `609 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.6484%`
- CLF residual > tol: `30.2527%`
- box residual > tol: `0.7434%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `10.8907%`
- CLF dominant: `87.1900%`
- box dominant: `1.9193%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `12.3153%`
- box dominant: `87.6847%`

Dominant family across all samples:

- hard CBF dominant: `54.4092%`
- soft CBF dominant: `9.3787%`
- CLF dominant: `35.3223%`
- box dominant: `0.8899%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.540138`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.250703`
- Min physical hard-barrier h when infeasible: `0.771866`
- Mean obstacle distance: `1.444620 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `1.425476`
- Mean D_t_curr: `1.605575`
- Mean D_t_floor: `0.143100`
- Mean D_t / |h| ratio: `1.087984`
- Mean D_t / actual-wind-only bound ratio: `0.245805`
- Mean D_t cap ratio current: `1.989526`
- Soft-clamp transition rate: `39.5007%`
- Hard-cap binding rate: `34.0039%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.5554%`
- CLF cost mean when active: `8.148935`
- CLF cost p95 when active: `20.629066`
- Mean t_clf: `6.564410`
- Max t_clf: `97.534828`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.566272 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4724%`
- Mean |psi1|: `12.374343`
- Mean |psi2|: `32.763302`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.446462`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.2176`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1889`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V17 Offline Gate

- Required feasibility rate > `97%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Required D_t / |h| in [1.3, 1.8]: `FAIL`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 25%: `FAIL`
- Required soft-clamp transition rate in [15%, 50%]: `PASS`
- Overall V17 math gate: `FAIL`
