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
- Wind max: `0.0500 N`
- DR scale: `0.2000`
- Actual wind acceleration scale: `1.8519 m/s^2`

## Feasibility

- Feasibility rate: `99.2273%`
- Infeasibility rate: `0.7727%`
- Infeasible samples: `633 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.6350%`
- CLF residual > tol: `30.1587%`
- box residual > tol: `0.7727%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `10.9661%`
- CLF dominant: `87.0254%`
- box dominant: `2.0086%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `12.0063%`
- box dominant: `87.9937%`

Dominant family across all samples:

- hard CBF dominant: `54.6472%`
- soft CBF dominant: `9.3079%`
- CLF dominant: `35.1331%`
- box dominant: `0.9119%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.573242`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `10.955745`
- Min physical hard-barrier h when infeasible: `0.795313`
- Mean obstacle distance: `1.450884 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `0.980684`
- Mean D_t_curr: `1.062439`
- Mean D_t_floor: `0.143715`
- Mean D_t wind term: `0.878260`
- Mean D_t mass term: `0.184179`
- Mean D_t parameter-rate term: `0.000000`
- Mean recent commanded u_xy norm: `5.846427`
- Mean D_t / |h| ratio: `0.763456`
- Mean D_t / actual-wind-only bound ratio: `0.170367`
- Mean D_t cap ratio current: `2.984509`
- Mean wind radial factor current: `0.547160`
- Soft-clamp transition rate: `18.1433%`
- Hard-cap binding rate: `15.2161%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.3943%`
- CLF cost mean when active: `8.107465`
- CLF cost p95 when active: `20.587656`
- Mean t_clf: `6.517938`
- Max t_clf: `94.866005`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.577636 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4724%`
- Mean |psi1|: `12.563478`
- Mean |psi2|: `33.309639`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.715193`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1943`
2. `Suspect 2: D_t over-conservative vs wind/h` score `0.1527`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V21 Offline Gate

- Required feasibility rate > `98%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Expected D_t / |h| target band [0.5, 0.7]: `OUT-OF-BAND`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 20%: `PASS`
- Soft-clamp transition diagnostic in [15%, 50%]: `IN-BAND`
- Overall V21 math gate: `PASS`
