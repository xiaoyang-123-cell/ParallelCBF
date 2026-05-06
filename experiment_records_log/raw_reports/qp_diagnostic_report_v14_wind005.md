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

- Feasibility rate: `99.2310%`
- Infeasibility rate: `0.7690%`
- Infeasible samples: `630 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.8218%`
- CLF residual > tol: `30.2991%`
- box residual > tol: `0.7690%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.4451%`
- CLF dominant: `86.5568%`
- box dominant: `1.9981%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `11.2698%`
- box dominant: `88.7302%`

Dominant family across all samples:

- hard CBF dominant: `54.3213%`
- soft CBF dominant: `9.6155%`
- CLF dominant: `35.1575%`
- box dominant: `0.9058%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.597062`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.208919`
- Min physical hard-barrier h when infeasible: `0.769832`
- Mean obstacle distance: `1.458519 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `1.781700`
- Mean D_t_curr: `1.781700`
- Mean D_t_floor: `0.144489`
- Mean D_t / |h| ratio: `1.564866`
- Mean D_t / actual-wind-only bound ratio: `0.330200`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol: `80.8252%`
- Mean t_clf: `6.694502`
- Max t_clf: `98.932968`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.571384 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4736%`
- Mean |psi1|: `12.634526`
- Mean |psi2|: `33.187962`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.492461`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.3130`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1898`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V13 Offline Gate

- Required feasibility rate > 97%: `PASS`
- Required box residual rate < 3%: `PASS`
- Required D_t / |h| around ~1.5, accepted [1.0, 2.0]: `PASS`
- Required CLF cost active rate substantially below old 94%, using < 75% as gate: `FAIL`
- Overall V13 math gate: `FAIL`
