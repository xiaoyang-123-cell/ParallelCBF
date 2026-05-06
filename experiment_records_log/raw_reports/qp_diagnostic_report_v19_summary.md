# QP Diagnostic Report V19 Summary

## V19 Patch Applied

- `SensorNoiseCfg.obstacle_rel_pos_std = 0.01`, mapped from the requested `obstacle_pos_sigma`.
- `wind_factor_values = [0.85, 0.70, 0.55, 0.30, 0.30]`.
- `D_t_cap_ratio_anchors = [3.0, 3.0, 3.5, 4.0, 4.5]`.
- `soft_clamp_smoothness = 0.2` unchanged.
- Mass perturbation was already at `mass_scale_range = (0.95, 1.05)` and QP `mass_scale_min/max = 0.95/1.05`.
- V19 diagnostic gates updated to high-wind `D_t / |h| in [1.6, 2.5]` and hard-cap binding `< 25%`.

## Layer 1 Results

| Metric | 0.05N Target | 0.05N Result | 0.20N Target | 0.20N Result |
| --- | ---: | ---: | ---: | ---: |
| Feasibility | > 97% | 99.2529% PASS | > 95% | 99.2444% PASS |
| Box residual rate | < 2% | 0.7471% PASS | < 2% | 0.7556% PASS |
| D_t / \|h\| | [1.3, 1.8] | 2.056705 FAIL | [1.6, 2.5] | 2.950837 FAIL |
| CLF cost p95 | < 30 | 20.720932 PASS | < 30 | 20.990263 PASS |
| Hard-cap binding | < 25% | 56.4709% FAIL | < 25% | 64.6448% FAIL |
| Soft-clamp transition | [15%, 50%] | 64.5947% FAIL | [15%, 50%] | 73.1860% FAIL |

## Verdict

V19 Layer 1 failed in both wind regimes. The failure is not marginal: three coupled metrics are outside target at both 0.05N and 0.20N. Feasibility, box residuals, and CLF p95 are healthy, but the robust margin remains too large relative to physical barrier distance and the soft cap is active too often.

The automatic V20 contingency was not applied because the directive authorized it only for a slight single-metric miss. This run has multi-metric, non-marginal misses, so the correct action is to stop before PPO and report the math failure.

## Artifacts

- Low wind report: `qp_diagnostic_report_v19_wind005.md`
- High wind report: `qp_diagnostic_report_v19_wind020.md`
