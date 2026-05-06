# QP Diagnostic Report V13

## Scope

RL hyperparameter tuning is suspended. This report verifies the V13 CLF demotion and D_t simplification patch at low wind.

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

- Feasibility rate: `93.2886%`
- Infeasibility rate: `6.7114%`
- Infeasible samples: `5498 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `10.2100%`
- CLF residual > tol: `46.6943%`
- box residual > tol: `6.7114%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `10.3016%`
- CLF dominant: `81.9491%`
- box dominant: `7.7493%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.6912%`
- CLF dominant: `36.9589%`
- box dominant: `62.3499%`

Dominant family across all samples:

- hard CBF dominant: `41.0132%`
- soft CBF dominant: `8.3728%`
- CLF dominant: `45.4639%`
- box dominant: `5.1501%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `1.632818`
- Min physical hard-barrier h over all samples: `-0.105248`
- Mean physical hard-barrier h when infeasible: `2.847638`
- Min physical hard-barrier h when infeasible: `0.004148`
- Mean obstacle distance: `1.229147 m`
- Min obstacle distance: `0.177909 m`
- Mean D_t used: `1.386533`
- Mean D_t_curr: `1.386533`
- Mean D_t_floor: `0.121768`
- Mean D_t / |h| ratio: `1.621689`
- Mean D_t / actual-wind-only bound ratio: `0.304559`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol: `80.6104%`
- Mean t_clf: `6.448743`
- Max t_clf: `104.122658`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `0.985754 m/s`
- Fraction with relative speed < 0.1 m/s: `1.0352%`
- Mean |psi1|: `7.287296`
- Mean |psi2|: `17.780354`
- Mean |psi2| at rel_speed < 0.1 m/s: `11.824170`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.3243`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.2365`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V13 Offline Gate

- Required feasibility rate > 97%: `FAIL`
- Required box residual rate < 3%: `FAIL`
- Required D_t / |h| around ~1.5, accepted [1.0, 2.0]: `PASS`
- Required CLF cost active rate substantially below old 94%, using < 75% as gate: `FAIL`
- Overall V13 math gate: `FAIL`
