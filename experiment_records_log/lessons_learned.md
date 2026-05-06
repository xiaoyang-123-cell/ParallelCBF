# Lessons Learned

## Architecture and Capacity

- Strong simple-task reward can be misleading. The early hover baseline looked excellent, but did not stress obstacle safety and ended with an abnormal policy std.
- A stable safety/QP layer does not imply task competence. V21/V22 repeatedly showed low collision/QP failure rates while long-horizon success remained poor.
- Small recurrent BC models can look stable while still underfitting large heterogeneous demonstration corpora. The V24 GRU floor at `0.0594598` was not a dataset-audit failure; it was a capacity bottleneck.
- The V25 causal Transformer immediately broke the GRU floor and reached `0.0060398` validation loss, so future BC anchors should treat context modeling capacity as a first-class design variable.

## Safety Semantics

- Missing telemetry is itself a gate failure. The first CBF v2 pilot abort was correct because CBF metrics were `N/A`; without observability, safety claims are theater.
- Offline QP diagnostics are the right response when PPO variants repeatedly hit the same safety bottleneck. V12-V20 prevented wasted training cycles.
- Collision, out-of-arena, success, and timeout must be separate termination reasons. A single `collision=True` flag can hide Layer-1 bugs and falsely implicate Layer 2.
- Dummy obstacle sentinels must never participate in physical collision accounting. The phantom collision episode is the canonical cautionary example.
- Layer-2 safety evidence should include barrier values near failure, not only binary outcomes. In the phantom collision case, positive `h_hard` values exposed the bug quickly.

## PPO and Critic Training

- Boundary exits cannot be treated only as reward shaping when they dominate the rollout distribution. The DualBarrier run created a fatal funnel: obstacle safety held, but arena exits consumed the learning signal.
- Value-loss plateau is a poor proxy for critic readiness. Attempt 2 had EV above `0.7` while value loss was still improving, so the actor stayed frozen unnecessarily.
- EV-based transition gates are better, but full-PPO handoff can still shock the optimizer. Gate V2 opened correctly, then forensics showed immediate `approx_kl ~= 1.87` and `clip_fraction=1.0`.
- PPO KL diagnostics are only as good as their rollout bookkeeping. Storing `old_log_prob=0` can masquerade as a policy KL explosion; `old_log_prob` must be recorded under the rollout policy for the nominal actor action.
- When a safety wrapper projects actions, PPO should be explicit about the semantic split: policy optimization stores the nominal actor action, while the environment executes the projected safe action and diagnostics store both.

## Observability and Watchdogs

- Watchdogs must fail closed on malformed metrics. Silent infinity/default fallbacks can blind safety or overfit rules.
- Watchdogs must be regime-aware. A value-loss slope rule that is sensible during critic warmup can become a false halt in FULL_PPO, where policy updates naturally reshape the value target distribution.
- Metrics should distinguish `nan because no episodes terminated this window` from `nan because numerical failure`. The Gate V2 resume produced `rollout/episode_return_std=nan` during no-termination windows while other safety metrics were finite.
- Training-time termination metrics are only meaningful if rollout collection preserves episode lifecycle. Resetting every 64-step context window in a 500-step environment can produce `termination_success=0` and `episode_return_mean=nan` even when safety and optimization are healthy.
- Final evaluation halt rules need numerical tolerances that match the semantics being claimed. A stopping-distance arena barrier at `-5e-08` is useful evidence, but it should be interpreted differently from obstacle collision or gross out-of-arena failure.
- Forensics dumps should capture the first bad transition after a phase change, including actor learning rate, KL, clip fraction, grad norm, value statistics, and phase name.

## Paper-Writing Hooks

- The clean paper narrative is not “one model solved everything.” It is a layered debugging story: BC capacity solved imitation loss; Layer-1 semantics fixed false collisions; Layer-2 arena promotion fixed boundary exits; PPO still exposed optimizer handoff risk.
- Negative results are evidence, not clutter. The DualBarrier diagnostic and Gate V1 halt justify TripleBarrier and Gate V2 respectively.
