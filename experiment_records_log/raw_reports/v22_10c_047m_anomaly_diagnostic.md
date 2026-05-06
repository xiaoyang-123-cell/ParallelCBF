# V22.10c 0.47m Anomaly Diagnostic

Status: Layer 3 held. Diagnostic protocol completed.

## Diagnostic 1: Curriculum Sync Check

At failed Layer 2 iteration `99`, the active goal-success radius is synchronized between termination/reward/logger.

| Location | Value / Evidence |
| --- | --- |
| Env termination logic | `_get_dones()` uses `goal_delta.norm() < self._goal_success_radius()` |
| Reward computation | `_get_rewards()` uses `goal_distance < self._goal_success_radius()` |
| Metrics logger | `task/goal_success_radius_current = 0.433666676283`, `task/goal_success_radius_mean = 0.433666676283` |
| Policy observation | No radius is included. Observation contains `goal_delta`, obstacle terms, alpha/gamma/rates/slack only. |

Conclusion: no reward/done/logger desync was found. The policy is not explicitly observing the annealed radius, but the environment-side scalar is synchronized.

Relevant failed-run scalar values at iter `99`:

| Metric | Value |
| --- | ---: |
| `task/goal_success_radius_current` | `0.433666676283 m` |
| `task/goal_success_radius_mean` | `0.433666676283 m` |
| `hover/goal_dist_mean` | `0.472833067179 m` |
| `task/goal_reached_rate` | `0.325012207031%` |
| `task/r_pbs_mean` | `0.298268556595` |

## Diagnostic 2: BC Pure Prior Check

Frozen BC actor was evaluated with no PPO updates and hardcoded `goal_radius = 0.4336666763`.

| Metric | Frozen BC Result |
| --- | ---: |
| Success rate | `77.999997%` |
| Final goal distance mean | `0.487148 m` |
| Min goal distance mean | `0.436461 m` |
| Min goal distance min | `0.433722 m` |
| Final step mean | `90.30` |
| Mean residual action norm | `0.051953` |

Conclusion: pure BC does not explain the PPO run parking at `0.4728 m`. The frozen prior can reach very close to the `0.4337 m` boundary and succeeds in about `78%` of episodes under that relaxed Layer-2 radius.

Interpretation: PPO updates appear to degrade or push the policy outward relative to the BC prior.

## Diagnostic 3: Timeout Saturation Check

The failed run did not primarily fail because episodes maxed out at the 20s horizon.

| Metric | Value |
| --- | ---: |
| Max possible episode length | `1000` steps (`20s / 0.02s`) |
| Iter 99 mean episode length | `173.60` |
| Tail-10 mean episode length | `125.14` |
| Frozen BC final step mean | `90.30` |

Conclusion: not a timeout saturation problem. Increasing max episode length to `300` steps would not address the observed failure; the current max is already `1000`, and failed-run episodes are ending far below that.

## Diagnostic 4: PPO Update Magnitude

Actor weights moved substantially from `model_0.pt` to `model_99.pt`.

| Weight Group | Relative L2 Change |
| --- | ---: |
| Actor MLP only | `14.2485%` |
| Actor MLP + log_std | `14.0648%` |
| Actor no normalizer count | `14.2524%` |
| Obs normalizer excluding count | `15.4673%` |

Per-layer relative changes:

| Tensor | Relative Change |
| --- | ---: |
| `mlp.0.weight` | `8.8979%` |
| `mlp.0.bias` | `11.9101%` |
| `mlp.2.weight` | `17.8326%` |
| `mlp.2.bias` | `22.4671%` |
| `mlp.4.weight` | `18.0349%` |
| `mlp.4.bias` | `17.3959%` |
| `distribution.log_std_param` | `7.2455%` |

Conclusion: PPO is updating the actor. The problem is not a frozen optimizer or no-op update path.

## Final Diagnosis

The `0.47 m` anomaly is not caused by:

- Curriculum desynchronization between reward, termination, and metrics.
- BC prior parking at `0.47 m`.
- Episode timeout saturation.
- PPO failing to update actor weights.

Most likely cause: **early PPO updates degrade the useful BC near-boundary behavior**. The frozen BC actor can graze the active `0.4337 m` boundary, but PPO moves the actor by about `14%` and the learned policy settles farther out at `0.4728 m`.

Immediate implication: do not launch Layer 3 from this checkpoint. The next fix should protect the BC prior during early PPO, likely via actor-freeze / low-LR warmup / KL-to-BC regularization / delayed residual updates, rather than changing CBF math or episode length.
