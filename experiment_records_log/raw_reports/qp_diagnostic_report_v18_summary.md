# V18 Layer 1 Offline QP Verification

## Scope

V18 replaces the static wind radial factor with a wind-dependent table:

- `wind_radial_anchors = [0.00, 0.05, 0.10, 0.20, 0.30]`
- `wind_factor_values = [0.85, 0.70, 0.55, 0.40, 0.35]`
- `D_t_cap_ratio_anchors = [2.5, 2.5, 3.0, 3.5, 4.0]`

The V17.5 stage-limit wind input fix and Softplus soft-min clamp are preserved.

No PPO training was launched.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v18_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v18_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 95% | 99.2737% | 99.2346% | PASS |
| Box residual rate | < 2% | 0.7263% | 0.7654% | PASS |
| D_t / \|h\| ratio | 0.05 N [1.3, 1.8], 0.20 N [1.8, 2.5] | 1.833818 | 2.859212 | FAIL |
| CLF cost p95 when active | < 30 | 20.683746 | 20.814573 | PASS |
| Hard-cap binding rate | < 30% | 64.6118% | 81.8188% | FAIL |
| Soft-clamp transition rate | [15%, 50%] | 73.2654% | 87.4097% | FAIL |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Mean D_t used | 3.001337 | 5.140052 |
| Mean D_t_curr | 4.422459 | 9.180483 |
| Mean D_t cap ratio current | 2.487122 | 3.481928 |
| Mean wind radial factor current | 0.696394 | 0.397934 |
| Infeasibility rate | 0.7263% | 0.7654% |

## Verdict

Layer 1: FAIL.

The adaptive radial factor is working numerically:

- 0.05 N factor: `0.696394`, near the intended 0.70.
- 0.20 N factor: `0.397934`, near the intended 0.40.

However, the D_t ratio remains above target at both wind levels:

- 0.05 N is slightly high at `1.833818`.
- 0.20 N is high at `2.859212`.

The clamp-rate gates also remain far above target, which indicates the current Softplus clamp and cap anchors are still operating over most samples rather than only near the transition boundary.

Per protocol, no PPO training was started.
