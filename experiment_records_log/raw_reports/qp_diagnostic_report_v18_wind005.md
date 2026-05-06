# QP Diagnostic Report V18

## Scope

RL hyperparameter tuning is suspended. This report verifies V18 sub-linear wind scaling with an adaptive radial factor and Softplus soft-min D_t clamping.

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

- Feasibility rate: `99.2737%`
- Infeasibility rate: `0.7263%`
- Infeasible samples: `595 / 81920`

## Binding Distribution

Violation-rate by constraint family:

- hard CBF residual > tol: `0.0000%`
- soft CBF residual > tol: `4.8633%`
- CLF residual > tol: `30.3430%`
- box residual > tol: `0.7263%`

Dominant family among infeasible/binding samples:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `11.3467%`
- CLF dominant: `86.7937%`
- box dominant: `1.8595%`

Dominant family among actual environment-infeasible samples only:

- hard CBF dominant: `0.0000%`
- soft CBF dominant: `0.0000%`
- CLF dominant: `12.6050%`
- box dominant: `87.3950%`

Dominant family across all samples:

- hard CBF dominant: `54.1016%`
- soft CBF dominant: `9.6753%`
- CLF dominant: `35.3394%`
- box dominant: `0.8838%`

## Physical Distance And D_t

- Mean physical hard-barrier h over all samples: `2.549356`
- Min physical hard-barrier h over all samples: `-0.102364`
- Mean physical hard-barrier h when infeasible: `11.346803`
- Min physical hard-barrier h when infeasible: `1.244683`
- Mean obstacle distance: `1.447084 m`
- Min obstacle distance: `0.185839 m`
- Mean D_t used: `3.001337`
- Mean D_t_curr: `4.422459`
- Mean D_t_floor: `0.143356`
- Mean D_t / |h| ratio: `1.833818`
- Mean D_t / actual-wind-only bound ratio: `0.461614`
- Mean D_t cap ratio current: `2.487122`
- Mean wind radial factor current: `0.696394`
- Soft-clamp transition rate: `73.2654%`
- Hard-cap binding rate: `64.6118%`

## CLF Cost Audit

- CLF cost active rate, t_clf > tol, diagnostic only: `80.7214%`
- CLF cost mean when active: `8.208492`
- CLF cost p95 when active: `20.683746`
- Mean t_clf: `6.626013`
- Max t_clf: `100.951019`
- Old V12 CLF residual active rate reference: `~94%`

## HOCBF Low-Speed Check

- Mean relative speed: `1.563161 m/s`
- Fraction with relative speed < 0.1 m/s: `0.4712%`
- Mean |psi1|: `12.399928`
- Mean |psi2|: `32.674469`
- Mean |psi2| at rel_speed < 0.1 m/s: `9.546367`

## Suspect Verdict

- Suspect 1 hard CBF dominates while h is large: `NO`
- Suspect 2 D_t / h or D_t / wind ratio huge: `NO`
- Suspect 3 low-speed HOCBF numerical fragility: `NO`

Additional finding: actual environment infeasibility matches box residuals, not hard CBF residuals. The direct mathematical wall is therefore the action box after the QP solve. The pressure source appears to be the robust/soft CBF and CLF system pushing the ADMM solution against or beyond the action box, with D_t oversized relative to physical h in the low-wind regime.

Ranked diagnosis:

1. `Suspect 2: D_t over-conservative vs wind/h` score `0.3668`
2. `Suspect 3: low-speed HOCBF numerical fragility` score `0.1909`
3. `Suspect 1: hard CBF / alpha too conservative` score `0.0000`

## Interpretation

The most important evidence is the binding distribution plus the physical h at infeasibility. If hard CBF dominates while h remains large and positive, the barrier gain/alpha pathway is too conservative. If D_t ratios dominate, the robust bound is still oversized relative to the low-wind regime. If neither dominates and psi2 spikes at near-zero relative speed, then the HOCBF recursion/curvature path is numerically fragile.

## V18 Offline Gate

- Required feasibility rate > `97%`: `PASS`
- Required box residual rate < 2%: `PASS`
- Required D_t / |h| in [1.3, 1.8]: `FAIL`
- Required CLF cost p95 when active < 30: `PASS`
- Required hard-cap binding < 30%: `FAIL`
- Required soft-clamp transition rate in [15%, 50%]: `FAIL`
- Overall V18 math gate: `FAIL`
