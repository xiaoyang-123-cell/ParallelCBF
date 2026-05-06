# V15 Layer 1 Offline QP Verification

## Scope

V15 applies radial wind scaling and an absolute D_t cap on top of the V13 CLF-as-cost architecture and V14 box expansion.

- `wind_radial_factor = 0.7`
- `D_t_max_ratio = 2.0`
- `rho_clf_cost = 8.0`
- `u_xy_min/max = +/-15.0`
- `u_z_min = -11.0`
- `u_z_max = 12.0`

No PPO training was launched. Layer 2 sanity and Layer 3 full training remain blocked unless all Layer 1 criteria pass.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v15_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v15_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 95% | 99.2432% | 99.2688% | PASS |
| Box residual rate | < 2% | 0.7568% | 0.7312% | PASS |
| D_t / \|h\| ratio | 0.05 N [1.3, 1.8], 0.20 N [1.8, 2.5] | 0.963256 | 1.574871 | FAIL |
| CLF cost p95 when active | < 30 | 20.611189 | 20.650513 | PASS |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Infeasibility rate | 0.7568% | 0.7312% |
| Mean D_t used | 1.276147 | 2.586461 |
| Mean D_t_curr | 1.449068 | 3.675939 |
| D_t cap active rate | 36.2073% | 74.9683% |
| CLF cost mean when active | 8.124710 | 8.181727 |
| Mean t_clf | 6.544101 | 6.588427 |
| Max t_clf | 98.140320 | 100.087486 |

## Verdict

Layer 1: FAIL.

V15 successfully fixes the false-flag CLF metric and preserves excellent QP feasibility. The action box is no longer the bottleneck, and CLF p95 is comfortably below the required threshold. However, the D_t cap/radial scaling combination is now too aggressive:

- 0.05 N D_t / |h| is `0.963256`, below the required `[1.3, 1.8]`.
- 0.20 N D_t / |h| is `1.574871`, below the required `[1.8, 2.5]`.
- The cap is active often, especially at 0.20 N (`74.9683%`), indicating the cap is suppressing the high-wind margin before it reaches the target band.

Per protocol, V15 sanity training and the 1500-iteration full run were not started.

## Targeted Retune Notes

The cleanest V16 retune is likely to raise the cap rather than undo the CLF-as-cost or box changes:

- Increase `D_t_max_ratio` above `2.0`, because the cap is active in 75% of 0.20 N samples.
- Recheck whether `wind_radial_factor = 0.7` should remain or move slightly upward only after cap recalibration.
- Keep CLF p95 as the gate metric; active rate remains diagnostic-only for a soft-cost formulation.
