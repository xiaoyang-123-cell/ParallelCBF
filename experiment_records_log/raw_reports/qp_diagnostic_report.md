# QP Feasibility Diagnostic Report

## Setup

- Canonical state: drone `(3.0, 0.0, 1.0)`, obstacle `(1.5, 0.05, 1.0)`, goal `(0.0, 0.0, 1.0)`.
- Initial velocity: zero. Safety radius: `0.370 m`.
- Mode A dynamics: controller mass `0.027 kg`, physics mass `0.02835 kg` (`scale=1.05`).
- Wind direction: `[-wind, 0, 0]`, i.e. toward the obstacle/goal line from the canonical start.
- Patch state: action box recalibrated to `u_xy in [-12, 12]`, `u_z in [-9, 10]`, yaw in `[-4, 4]`; CBF uses squared-distance HOCBF with kinematic curvature; robust `D_t` includes wind, mass mismatch, perception, obstacle velocity, and parameter-rate terms.
- Heatmap value: number of survived open-loop QP steps out of 100 using the same infeasibility semantics as training: hard CBF violation OR ReLUQP auxiliary/box violation aborts the cell.
- `structural` success counts are shown separately and mean only the exact hard-CBF-with-box feasibility test passed; soft CBF and CLF are slack-relaxed and therefore not structural blockers.

## Probe 1: Feasibility Heatmaps

### Wind 0.0 N

- Solver-aware success cells: `961` / `961`.
- Structural hard-CBF success cells: `961` / `961`.
- Max solver-aware survived steps: `100`.
- Max structural survived steps: `100`.
- Best min distance: `1.5012 m`.
- First solver-aware infeasible step range: `-1` to `-1` (`-1` means no solver-aware failure).
- Max ReLUQP auxiliary/box gap: `0.000000e+00`.

![heatmap wind 0.0 N](logs/qp_diagnostics/heatmap_wind_0p0.png)

### Wind 0.1 N

- Solver-aware success cells: `961` / `961`.
- Structural hard-CBF success cells: `961` / `961`.
- Max solver-aware survived steps: `100`.
- Max structural survived steps: `100`.
- Best min distance: `1.5012 m`.
- First solver-aware infeasible step range: `-1` to `-1` (`-1` means no solver-aware failure).
- Max ReLUQP auxiliary/box gap: `0.000000e+00`.

![heatmap wind 0.1 N](logs/qp_diagnostics/heatmap_wind_0p1.png)

### Wind 0.2 N

- Solver-aware success cells: `961` / `961`.
- Structural hard-CBF success cells: `961` / `961`.
- Max solver-aware survived steps: `100`.
- Max structural survived steps: `100`.
- Best min distance: `1.5012 m`.
- First solver-aware infeasible step range: `-1` to `-1` (`-1` means no solver-aware failure).
- Max ReLUQP auxiliary/box gap: `0.000000e+00`.

![heatmap wind 0.2 N](logs/qp_diagnostics/heatmap_wind_0p2.png)

### Wind 0.3 N

- Solver-aware success cells: `736` / `961`.
- Structural hard-CBF success cells: `961` / `961`.
- Max solver-aware survived steps: `100`.
- Max structural survived steps: `100`.
- Best min distance: `1.5012 m`.
- First solver-aware infeasible step range: `0` to `90` (`-1` means no solver-aware failure).
- Max ReLUQP auxiliary/box gap: `3.828573e-01`.

![heatmap wind 0.3 N](logs/qp_diagnostics/heatmap_wind_0p3.png)

Answer: under `0.3 N` wind, any alpha/gamma pair works under training's solver-aware criterion? **YES**.
Answer: under `0.3 N` wind, majority feasible? **YES** (`736` / `961` = `0.766`).
Answer: under `0.3 N` wind, any alpha/gamma pair is structurally hard-CBF feasible if solver auxiliary violations are ignored? **YES**.

## Probe 2: Disturbance Bound D_t Check

| wind N | gamma | D_t | actual accel disturbance | actual hddot disturbance | D_t / actual hddot |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.1 | 36.524208 | 0.260774 | 0.782755 | 46.661090 |
| 0.0 | 0.5 | 36.512211 | 0.260774 | 0.782755 | 46.645763 |
| 0.0 | 1.0 | 36.497208 | 0.260774 | 0.782755 | 46.626596 |
| 0.0 | 2.0 | 36.467209 | 0.260774 | 0.782755 | 46.588271 |
| 0.1 | 0.1 | 36.524208 | 3.269201 | 9.813050 | 3.722004 |
| 0.1 | 0.5 | 36.512211 | 3.269201 | 9.813050 | 3.720781 |
| 0.1 | 1.0 | 36.497208 | 3.269201 | 9.813050 | 3.719252 |
| 0.1 | 2.0 | 36.467209 | 3.269201 | 9.813050 | 3.716195 |
| 0.2 | 0.1 | 36.524208 | 6.796437 | 20.400637 | 1.790346 |
| 0.2 | 0.5 | 36.512211 | 6.796437 | 20.400637 | 1.789758 |
| 0.2 | 1.0 | 36.497208 | 6.796437 | 20.400637 | 1.789023 |
| 0.2 | 2.0 | 36.467209 | 6.796437 | 20.400637 | 1.787552 |
| 0.3 | 0.1 | 36.524208 | 10.323742 | 30.988428 | 1.178640 |
| 0.3 | 0.5 | 36.512211 | 10.323742 | 30.988428 | 1.178253 |
| 0.3 | 1.0 | 36.497208 | 10.323742 | 30.988428 | 1.177769 |
| 0.3 | 2.0 | 36.467209 | 10.323742 | 30.988428 | 1.176801 |

Answer: `D_t` at `wind=0.3 N, gamma=0.5` is `36.512`; wind-0.3 sweep range is `36.467` to `36.524`.
For the key case `wind=0.3 N, gamma=0.5`, `D_t / actual_hddot` ratio is `1.1783`.
Max `D_t / actual_hddot` across this check is `46.661`.

## Probe 3: HOCBF Recursion Check

- Canonical zero-velocity error: `0.000000e+00`.
- Max open-loop error over 100 DR steps: `0.000000e+00`.

Answer: recursive vs direct HOCBF discrepancy `< 1e-6`? **YES**.

## Probe 4: Slack/Bias Audit

- CSV: `logs/qp_diagnostics/slack_bias_audit.csv`.
- Plot: `logs/qp_diagnostics/slack_bias_audit.png`.
- `max |u_nom_y|`: `1.394797`.
- `max |u_safe_y|`: `1.117704`.
- `max |u_safe_clipped_y|`: `1.117704`.
- `max delta_cbf_soft`: `0.017315`.
- `max delta_clf`: `12.963546`.
- Minimum obstacle distance: `0.893503 m`.
- First hard infeasible step: `-1.0`.
- First ReLUQP auxiliary/box infeasible step: `-1.0`.

![slack bias audit](logs/qp_diagnostics/slack_bias_audit.png)

## Ranked Diagnosis

1. **Patched feasibility status is determined by solver-aware success, not structural-only success.** Structural success can still hide ReLUQP box/auxiliary residuals.
2. **The action box now exposes the requested 45-degree tilt envelope.** This removes the obvious `0.3 N` wind authority mismatch caused by the previous `u_xy=4` cap.
3. **The robust disturbance term is now scaled in squared-distance HOCBF units.** This is why `D_t` should be compared against `grad_h * acceleration_disturbance`, not raw acceleration disturbance.
4. **The HOCBF recurrence is verified against the direct squared-distance second derivative.** Any remaining infeasibility is therefore more likely backend/numerical or policy/environment, not this curvature bug.

## Offline Pass Criteria

- Majority feasible at `0.3 N`: `PASS`.
- `D_t ~= 35`: `PASS`.
- HOCBF discrepancy `< 1e-6`: `PASS`.
