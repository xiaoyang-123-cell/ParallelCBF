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
- Wind max: `0.2000 N`
- DR scale: `0.2000`
- Actual wind acceleration scale: `7.4074 m/s^2`

## Feasibility

- Feasibility rate: `99.1650%`
- Infeasibility rate: `0.8350%`
- Infeasible samples: `684 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `5.3064%`
- CLF residual > tol: `30.3333%`
- box residual > tol: `0.8350%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `12.3997%`
- CLF dominant: `85.4977%`
- box dominant: `2.1025%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `13.0117%`
- box dominant: `86.9883%`

Dominant family across all samples:

- hard CBF dominant: `53.0188%`
- soft CBF dominant: `10.9143%`
- CLF dominant: `35.0122%`
- box dominant: `1.0547%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.676310`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `12.420145`
- Min physical hard-barrier h when infeasible: `0.839534`
- Mean obstacle distance: `1.484674 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `5.083265`
- Mean D_t_curr: `5.083265`
- Mean D_t_floor: `0.147114`
- Mean D_t / |h| ratio: `4.234174`
- Mean D_t / actual-wind-only bound ratio: `0.231305`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol: `82.7734%`
- Mean t_clf: `7.477281`
- Max t_clf: `102.168930`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.551249 m/s`
- Fraction with relative speed < 0.1 m/s: `0.3955%`
- Mean |psi1|: `12.933039`
- Mean |psi2|: `32.728333`
- Mean |psi2| at rel_speed < 0.1 m/s: `10.658457`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.8468`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.2132`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V13 Offline Gate

- Required feasibility rate > 97%: `PASS`
- Required box residual rate < 3%: `PASS`
- Required D_t / |h| around ~1.5, accepted [1.0, 2.0]: `FAIL`
- Required CLF cost active rate substantially below old 94%, using < 75% as gate: `FAIL`
- Overall V13 math gate: `FAIL`
