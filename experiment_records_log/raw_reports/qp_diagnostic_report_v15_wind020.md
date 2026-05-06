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

- Feasibility rate: `99.2688%`
- Infeasibility rate: `0.7312%`
- Infeasible samples: `599 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.7595%`
- CLF residual > tol: `30.2869%`
- box residual > tol: `0.7312%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.1693%`
- CLF dominant: `86.9362%`
- box dominant: `1.8944%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `11.8531%`
- box dominant: `88.1469%`

Dominant family across all samples:

- hard CBF dominant: `54.1992%`
- soft CBF dominant: `9.5911%`
- CLF dominant: `35.3357%`
- box dominant: `0.8740%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.545149`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.244473`
- Min physical hard-barrier h when infeasible: `1.244297`
- Mean obstacle distance: `1.446010 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `2.586461`
- Mean D_t_curr: `3.675939`
- Mean D_t_floor: `0.143231`
- Mean D_t / |h| ratio: `1.574871`
- Mean D_t / actual-wind-only bound ratio: `0.098766`
- D_t cap active rate: `74.9683%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.5261%`
- CLF cost mean when active: `8.181727`
- CLF cost p95 when active: `20.650513`
- Mean t_clf: `6.588427`
- Max t_clf: `100.087486`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.569279 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4736%`
- Mean |psi1|: `12.399771`
- Mean |psi2|: `32.804123`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.375492`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.3150`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1875`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V15 Offline Gate

- Required feasibility rate > `95%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Required D_t / |h| in [1.8, 2.5]: `FAIL`
- Required CLF cost p95 when active < 30: `PASS`
- Overall V15 math gate: `FAIL`
