# QP Diagnostic Report V12

## Scope

RL hyperparameter tuning is suspended. This report verifies the V12 dynamic D_t patch at low wind.

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

- Feasibility rate: `92.3340%`
- Infeasibility rate: `7.6660%`
- Infeasible samples: `6280 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `28.2166%`
- CLF residual > tol: `91.7773%`
- box residual > tol: `7.6660%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.2243%`
- CLF dominant: `98.5576%`
- box dominant: `1.2182%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `85.3822%`
- box dominant: `14.6178%`

Dominant family across all samples:

- hard CBF dominant: `5.1184%`
- soft CBF dominant: `0.7825%`
- CLF dominant: `92.9724%`
- box dominant: `1.1267%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `1.354804`
- Min physical hard-barrier h over all samples: `-0.105248`
- Mean physical hard-barrier h when infeasible: `2.385515`
- Min physical hard-barrier h when infeasible: `0.017385`
- Mean obstacle distance: `1.138094 m`
- Min obstacle distance: `0.177909 m`
- Mean D_t used: `1.630268`
- Mean D_t_curr: `1.278634`
- Mean D_t_obs: `0.162071`
- Mean D_t_floor: `0.112292`
- Mean D_t / |h| ratio: `2.220204`
- Mean D_t / actual-wind-only bound ratio: `0.376556`
- Mean observed disturbance EMA: `0.068171 m/s^2`

## HOCBF Low-Speed Check

- Mean relative speed: `0.872442 m/s`
- Fraction with relative speed < 0.1 m/s: `1.0852%`
- Mean |psi1|: `6.030835`
- Mean |psi2|: `14.728004`
- Mean |psi2| at rel_speed < 0.1 m/s: `12.270888`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.4440`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.2454`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V12 Offline Gate

- Required D_t / |h| < 2.0: `FAIL`
- Required feasibility rate > 95%: `FAIL`
- Required binding_box_rate drop dramatically, using < 2% as the mechanical gate: `FAIL`
- Overall V12 math gate: `FAIL`
