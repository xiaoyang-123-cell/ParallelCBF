# V12-V20 Layer-1 QP Diagnostics

Status: `FAIL`

## Purpose

Diagnose and tune the Layer-1 CBF/QP math before allowing more PPO training. These runs use reconstructed V11 rollout states and manually evaluate the QP under low/high wind.

## Setup

- Reference run: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10`
- Reference checkpoint: `model_99.pt`
- Reports: `qp_diagnostic_report_v12.md` through `qp_diagnostic_report_v20_summary.md`
- Wind reports: `qp_diagnostic_report_v14_wind005.md` through `qp_diagnostic_report_v20_wind020.md`

## Result

- V12 dynamic `D_t` patch reduced mean `D_t / |h|` to `2.220204`, but feasibility was only `92.334%`; gate failed.
- V13 CLF demotion and simplified `D_t` improved mean `D_t / |h|` to `1.621689`, but feasibility still failed and CLF cost active rate remained too high.
- V14 achieved high feasibility at both winds (`99.2310%` at 0.05N, `99.1650%` at 0.20N) and box residuals below 1%, but high-wind `D_t / |h|=4.234174` and CLF cost active rate still failed.
- V15/V16 preserved excellent feasibility but could not satisfy both low-wind and high-wind `D_t` ratio/cap gates simultaneously.
- V17 introduced adaptive cap ratios, but a wind-reference bug kept cap ratio near `~2.0` even in the 0.20N diagnostic.
- V17.5 fixed the stage-limit wind input; cap tracking worked, but high-wind `D_t / |h|=3.311014`, hard-cap binding, and soft-clamp rates failed.
- V18/V19 tried wind-dependent radial factors and lean shield calibration; feasibility stayed above `99%`, but `D_t` and cap metrics stayed out of band.
- V20 removed two phantoms: mass term fell to about `0.18` and parameter-rate term became exactly `0.0`, but wind term dominated `D_t`; V20 Layer 1 still failed.

## Diagnosis

The major scientific lesson is that feasibility alone was not enough. From V14 onward the QP often looked numerically feasible, but robust-margin calibration remained too conservative or cap-heavy. V20 proved the mass and alpha-rate phantoms were not the remaining blocker; wind/attitude-lag modeling had to be handled more explicitly.

## Decision

Do not launch full PPO from V12-V20. Continue to V21 with drag/wind compensation and lock the QP math only after both offline and sanity gates pass.

## Evidence

- `qp_diagnostic_report_v12.md`
- `qp_diagnostic_report_v13.md`
- `qp_diagnostic_report_v14_summary.md`
- `qp_diagnostic_report_v15_summary.md`
- `qp_diagnostic_report_v16_summary.md`
- `qp_diagnostic_report_v17_summary.md`
- `qp_diagnostic_report_v17_5_summary.md`
- `qp_diagnostic_report_v18_summary.md`
- `qp_diagnostic_report_v19_summary.md`
- `qp_diagnostic_report_v20_summary.md`

