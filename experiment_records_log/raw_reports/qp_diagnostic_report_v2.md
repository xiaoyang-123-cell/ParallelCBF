# QP Diagnostic Report V2

## Scope

RL hyperparameter tuning is suspended. This report inspects the CBF/QP bottleneck at low wind.

Important limitation: the V11 TensorBoard run stores scalar telemetry and checkpoints, but not full per-step state tensors. This diagnostic therefore reconstructs a deterministic low-wind rollout from `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt` using the V11 environment configuration, then manually evaluates the QP over the states visited in that rollout.

## Setup

- Run source: `/data/uav_project/ParallelCBF/logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10`
- Checkpoint: `model_99.pt`
- Envs: `512`
- Steps: `160`
- Samples: `81920`
- Wind max: `0.0500 N`
- Actual wind acceleration scale: `1.8519 m/s^2`

## Feasibility

- Feasibility rate: `91.8042%`
- Infeasibility rate: `8.1958%`
- Infeasible samples: `6714 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `37.7649%`
- CLF residual > tol: `94.1089%`
- box residual > tol: `8.1958%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.1929%`
- CLF dominant: `98.8141%`
- box dominant: `0.9930%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `88.5761%`
- box dominant: `11.4239%`

Dominant family across all samples:

- hard CBF dominant: `3.8843%`
- soft CBF dominant: `0.4321%`
- CLF dominant: `94.7473%`
- box dominant: `0.9363%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `1.416013`
- Min physical hard-barrier h over all samples: `-0.106857`
- Mean physical hard-barrier h when infeasible: `2.297500`
- Min physical hard-barrier h when infeasible: `0.021146`
- Mean obstacle distance: `1.165870 m`
- Min obstacle distance: `0.173330 m`
- Mean D_t used: `7.000934`
- Mean D_t / |h| ratio: `8.897154`
- Mean D_t / actual-wind-only bound ratio: `1.611083`

## HOCBF Low-Speed Check

- Mean relative speed: `0.818521 m/s`
- Fraction with relative speed < 0.1 m/s: `1.1230%`
- Mean |psi1|: `6.294759`
- Mean |psi2|: `14.598282`
- Mean |psi2| at rel_speed < 0.1 m/s: `12.775754`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `YES`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `1.7794`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.2555`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.
