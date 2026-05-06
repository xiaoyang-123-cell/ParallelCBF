# V23.0 Phase 2 Integration Smoke Report

Status: PASS. A*, GRU, waypoint lookahead, V22.10 CBF-QP, and the 50 Hz environment loop are now wired together and survived the 50-iteration smoke test.

No BC re-collection or Phase 3/Layer 3 launch was performed.

## Integration Implemented

- `env.reset()` now builds reset-time V23 waypoint paths from sampled start/goal/obstacle state.
- The planner path is stored in vectorized waypoint buffers and advanced online.
- `env.step()` now routes the GRU residual through the V23 hierarchy:

```text
GRU residual -> u_residual
waypoint manager -> effective_target, blend_ratio
P-PD nominal target = effective_target
u_combined = u_nominal + lambda_residual * u_residual
V22.10 CBF-QP -> u_safe
low-level controller -> rotor thrusts
```

- V23 remains gated behind `--v23_gru_policy`; the default V22 MLP path is not converted to GRU.
- The V23 observation schema remains `26` dimensions:

```text
state_18 + wp_curr(3) + wp_next(3) + progress_to_curr(1) + blend_ratio(1)
```

## Bug Fixed During Smoke

The first compile-level run stopped inside `RslRlVecEnvWrapper(env)`, before runner construction. The wrapper reset exposed a PyTorch API bug in `_update_waypoint_manager()`:

```text
TypeError: clamp() received an invalid combination of arguments - got (max=Tensor, min=int)
```

Fix: replaced `self._wp_current_index.clamp(min=0, max=valid_last_index)` with:

```python
torch.minimum(self._wp_current_index.clamp_min(0), valid_last_index)
```

After the fix, the env wrapper, recurrent runner, GRU actor/critic construction, and PPO learn path all returned normally.

## 50-Iteration Smoke Test

Run:

```text
logs/rsl_rl/uav_safe_3B1_v23_phase2_smoke/2026-04-28_23-25-51_layer2_50iter
```

Command:

```bash
PYTHONUNBUFFERED=1 timeout 1800s conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/train_ppo.py \
  --num_envs 512 \
  --seed 231 \
  --max_iterations 50 \
  --cbf_enabled True \
  --v23_gru_policy \
  --v22_9_residual_overrides \
  --experiment_name uav_safe_3B1_v23_phase2_smoke \
  --run_name layer2_50iter \
  --headless
```

Result: completed normally at iteration `49/50`. No NaN check failed, no crash occurred, and `runner.learn` returned normally.

## Final Iteration 49 Metrics

| Metric | Value |
| --- | ---: |
| `Train/mean_reward` | 17.0948 |
| `Train/mean_episode_length` | 38.4800 |
| `Policy/mean_std` | 0.3628 |
| `Loss/clip_fraction` | 0.0381 |
| `task/episode_success_rate` | 100.0000% |
| `task/final_dist_mean_successful` | 0.4484 m |
| `task/goal_reached_instantaneous_rate` | 2.4841% |
| `task/goal_curriculum_radius` | 0.5000 m |
| `task/goal_success_radius_current` | 0.4500 m |
| `task/obstacle_presence_prob` | 0.0000 |
| `task/obstacle_present_rate` | 0.0000 |
| `task/residual_lambda_current` | 0.3429 |
| `action/residual_norm_mean` | 1.8073 |
| `action/nominal_accel_norm_mean` | 1.3088 |
| `action/combined_accel_norm_mean` | 1.3019 |
| `task/waypoint_advances_per_episode` | 0.0000 |
| `task/mean_blend_ratio` | 0.0000 |
| `task/waypoint_advance_rate` | 0.0000 |
| `task/waypoint_index_mean` | 1.0000 |
| `safety/collision_rate` | 0.0000% |
| `safety/fallback_activation_rate` | 0.0000% |
| `safety/transient_infeasibility_rate` | 0.0000% |
| `qp/hard_cap_binding_rate` | 0.0000% |
| `qp/infeasibility_rate` | 0.0000% |
| `dr/wind_max_current` | 0.0500 N |

Waypoint note: early V23 curriculum has an empty room and direct two-point paths. The manager advances to the final waypoint immediately, so `waypoint_index_mean = 1.0`; no multi-waypoint detours are expected yet.

## Curriculum Self-Test

Command:

```bash
conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v23_phase2_integration_self_test.py
```

Result:

```text
V23 Phase 2 integration self-test PASS
[planner_and_blend] {'path_len': 2, 'first': [-0.6, 0.0, 1.0], 'last': [0.6, 0.0, 1.0], 'mid_blend': 0.1666666865348816, 'near_blend': 0.4666668176651001}
[curriculum_schedule] {'schedule': {0: (0.5, 0.0, 0.05), 300: (0.5, 0.0, 0.05), 800: (1.5, 0.42857142857142855, 0.05), 1200: (2.071428571428571, 1.0, 0.13571428571428573), 1500: (2.5, 1.0, 0.2)}}
[source_integration_contract] {'contracts': 5}
```

## GRU Regression

Command:

```bash
conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v23_stream2_gru_unit_tests.py
```

Result: PASS for all 7 pitfall tests.

```text
PASS | Hidden State Reset
PASS | Truncation vs. Termination
PASS | Sequence Batching
PASS | BC Init
PASS | Log Std Stability
PASS | Obs Dimension
PASS | Inference Latency
V23 Stream 2 GRU pitfall tests PASS
```

Latest latency sample:

```text
mean_ms=0.3833, p95_ms=0.5535
```

## V22 CBF Regression

Pytest command:

```bash
PYTHONPATH=/data/uav_project/ParallelCBF PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run --no-capture-output -n parallel_uav pytest -q \
  ParallelCBF_UAV/tests/test_dual_barrier_qp_regression.py \
  ParallelCBF_UAV/tests/test_qp_regression.py
```

Result:

```text
2 passed in 1.27s
```

V22.10 Layer 1 diagnostic:

```bash
conda run --no-capture-output -n parallel_uav \
  python scripts/qp_v22_10_attitude_lag_diagnostic.py \
  --device cuda \
  --batch_size 2048 \
  --report qp_diagnostic_report_v23_phase2_v22_10_regression_rerun.md
```

Result:

| Scenario | Hard Cap | Target | Feasibility | Result |
| --- | ---: | ---: | ---: | --- |
| Hover, low wind | 13.87% | <15.00% | 100.00% | PASS |
| Approach, high wind | 12.01% | <35.00% | 100.00% | PASS |
| Aggressive, high wind | 36.28% | <50.00% | 100.00% | PASS |

## Verdict

Day 3/4 Phase 2 integration gate is passed. The closed-loop V23 stack runs at 50 Hz under PPO collection/update, GRU recurrent rollout is stable, hidden-state behavior remains covered by the 7 pitfall tests, V23 staggered curriculum self-test passes, and V22.10 CBF math regressions still pass.

Authorized stopping point reached: do not proceed to BC re-collection until explicit approval.
