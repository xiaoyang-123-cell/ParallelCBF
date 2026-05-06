# QP Diagnostic Report V19

## Scope

RL hyperparameter tuning is suspended. This report verifies the V19 Lean Shield calibration with slimmed obstacle observation noise, adaptive radial wind factors, and graduated Softplus D_t caps.

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

- Feasibility rate: `99.2444%`
- Infeasibility rate: `0.7556%`
- Infeasible samples: `619 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.9658%`
- CLF residual > tol: `30.4517%`
- box residual > tol: `0.7556%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.7183%`
- CLF dominant: `86.3417%`
- box dominant: `1.9400%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `11.7932%`
- box dominant: `88.2068%`

Dominant family across all samples:

- hard CBF dominant: `53.3752%`
- soft CBF dominant: `10.3406%`
- CLF dominant: `35.4004%`
- box dominant: `0.8838%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.538532`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.533784`
- Min physical hard-barrier h when infeasible: `0.165787`
- Mean obstacle distance: `1.447398 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `4.801456`
- Mean D_t_curr: `7.078574`
- Mean D_t_floor: `0.143384`
- Mean D_t / |h| ratio: `2.950837`
- Mean D_t / actual-wind-only bound ratio: `0.184886`
- Mean D_t cap ratio current: `3.979687`
- Mean wind radial factor current: `0.298477`
- Soft-clamp transition rate: `73.1860%`
- Hard-cap binding rate: `64.6448%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.9973%`
- CLF cost mean when active: `8.328768`
- CLF cost p95 when active: `20.990263`
- Mean t_clf: `6.746078`
- Max t_clf: `103.775024`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.556839 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4871%`
- Mean |psi1|: `12.331757`
- Mean |psi2|: `32.363586`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.679530`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.5902`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1936`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V19 Offline Gate

- Required feasibility rate > `95%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Required D_t / |h| in [1.6, 2.5]: `FAIL`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 25%: `FAIL`
- Required soft-clamp transition rate in [15%, 50%]: `FAIL`
- Overall V19 math gate: `FAIL`
