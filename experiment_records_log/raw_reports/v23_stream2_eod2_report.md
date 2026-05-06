# V23.0 Stream 2 EOD 2 Report

Status: PASS. Day 1 red flags were corrected, V23 GRU standalone path was added, and all 7 GRU pitfall tests passed.

## Part 1: Day 1 Clarifications

| Question | Original State | Action Taken | Final Status |
| --- | --- | --- | --- |
| Collinearity tolerance | Used exact integer direction equality | Replaced with floating cross-product tolerance: `cross_norm <= 1e-6 * scale` | PASS |
| 3D vs 2D A* | Planner was 2D / 8-connected | Upgraded to 3D / 26-connected A* with 2D grid compatibility as a single z-layer | PASS |

Day 1 standalone test was re-run after the fix:

```text
V23 Stream 1 standalone tests PASS

[open_grid]
  raw_path: [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
  smoothed_path: [(0, 0, 0), (5, 5, 5)]
  cost_cells: 8.660254037844386
  cost_meters: 1.7320508075688772

[obstacle_detour]
  raw_path: [(0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2), (6, 2, 2)]
  smoothed_path: [(0, 2, 2), (6, 2, 2)]
  path_len_cells: 7

[waypoint_blending]
  blend_ratios: [0.0, 0.0, 0.25, 0.5, 0.75, 1.0]
```

## Stream 2 Implementation

Files changed:

- `ParallelCBF_UAV/planning/grid_planner.py`
- `ParallelCBF_UAV/tools/v23_stream1_standalone_test.py`
- `ParallelCBF_UAV/envs/uav_safe_env.py`
- `ParallelCBF_UAV/training/ppo_cfg.py`
- `ParallelCBF_UAV/train_ppo.py`
- `ParallelCBF_UAV/tools/v23_stream2_gru_unit_tests.py`

Implemented as a V23-only path:

- Added `--v23_gru_policy`.
- Default V22 path remains MLP and unchanged.
- V23 actor/critic use rsl_rl native `RNNModel`:
  - GRU hidden dim: `128`
  - MLP head: `[64]`
  - actor Gaussian std init: `0.35`
- Added V23 waypoint observation mode:
  - `state_18`
  - `wp_curr`
  - `wp_next`
  - `progress_to_curr`
  - `blend_ratio`

Important note: the requested "21-dim" schema conflicts with 3D waypoint fields. Since V23 navigation is now explicitly 3D, the safe schema is `18 + 3 + 3 + 1 + 1 = 26`. I implemented `26` to avoid silently dropping Z information.

## 7 GRU Pitfall Unit Tests

Command:

```bash
conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v23_stream2_gru_unit_tests.py
```

Output:

```text
V23 Stream 2 GRU pitfall tests
PASS | Hidden State Reset: {'done_hidden_abs_max': 0.0, 'live_hidden_norm': 0.47003674507141113}
PASS | Truncation vs. Termination: {'bootstrap_truncations': True, 'is_finite_horizon': False, 'time_outs_wrapper': 'RslRlVecEnvWrapper'}
PASS | Sequence Batching: {'padded_obs_shape': (6, 3, 26), 'actor_hidden_shape': (1, 3, 128), 'mask_true_count': 12}
PASS | BC Init: {'bc_h0_abs_max': 0.0, 'mean_output_shape': (8, 3)}
PASS | Log Std Stability: {'std_min': 0.34998682141304016, 'std_max': 0.35006991028785706}
PASS | Obs Dimension: {'obs_dim': 26, 'terms': {'state_18': 18, 'wp_curr': 3, 'wp_next': 3, 'progress_to_curr': 1, 'blend_ratio': 1}, 'storage_shape': (3, 2, 26)}
PASS | Inference Latency: {'mean_ms': 0.461043615014205, 'p95_ms': 0.5985710013192147}
V23 Stream 2 GRU pitfall tests PASS
```

## Isaac Compile Check

Command:

```bash
timeout 360s conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/train_ppo.py \
  --num_envs 16 \
  --seed 23 \
  --max_iterations 0 \
  --cbf_enabled True \
  --v23_gru_policy \
  --experiment_name uav_safe_3B1_v23_gru_compile_check \
  --run_name stream2_forward \
  --headless
```

Result: PASS, exit code `0`.

Run:

```text
logs/rsl_rl/uav_safe_3B1_v23_gru_compile_check/2026-04-28_21-50-11_stream2_forward
```

Config evidence:

```text
agent.yaml:
  actor.class_name: RNNModel
  actor.rnn_type: gru
  actor.rnn_hidden_dim: 128
  actor.hidden_dims: [64]
  critic.class_name: RNNModel
  critic.rnn_type: gru
  critic.rnn_hidden_dim: 128

env.yaml:
  observation_space: 26
  num_observations: 26
  v23_waypoint_observation_enabled: true
  v23_waypoint_observation_dim: 26
```

The Isaac startup emitted known extension warnings around `omni.kit.test`/`CXXABI`, but they did not stop the run and the process exited normally.

## EOD 2 Verdict

Stream 2 standalone is complete. The GRU path compiles, recurrent hidden states reset correctly, timeout bootstrap is preserved, rollout sequence batching carries hidden states, BC sequence starts use zero hidden states, std remains stable through a PPO update, the rollout storage/normalizer use the new observation dimension, and batch=1 inference is comfortably faster than 50Hz.

No Phase 2 integration or PPO training was launched.
