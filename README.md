# ParallelCBF

Composable safety filters, deterministic test fixtures, and auditability tools
for parallel robot-learning pipelines.

While Isaac Lab provides massive parallel UAV simulation, OmniSafe and safe-control-gym provide constrained-RL benchmarks, and CBFKit provides control-barrier-function synthesis tooling, no existing framework unifies these capabilities for end-to-end safety-constrained training. ParallelCBF is the first framework to unify (i) tensor-parallel UAV environments, (ii) hard-gate CBF safety filters, (iii) sharded BC-to-RL pipelines, and (iv) first-class operational auditability — pre-registration, watchdog registries, failure forensics, and dataset audits as composable APIs rather than user-implemented scripts.

## Why ParallelCBF?

ParallelCBF is a small, simulator-agnostic framework for composing:

- safety-aware environments;
- control barrier function safety filters;
- SB3/CleanRL-style algorithms;
- watchdogs, pre-registration artifacts, atomic checkpoints, and failure
  forensics.

The v0.1 scope is intentionally narrow: pure Python/NumPy/PyTorch CPU examples,
strict type hints, property tests, and reusable contracts. Isaac Lab adapters,
Mamba policies, and PPO training integrations are deferred to v0.2.

## Capability Comparison

| Capability | ParallelCBF | Isaac Lab | OmniSafe | safe-control-gym | CBFKit |
| --- | --- | --- | --- | --- | --- |
| Simulator-agnostic safety API | Yes | Partial | Partial | Partial | Partial |
| Extends Gymnasium-style envs | Yes | Yes | Yes | Yes | No |
| Composable safety wrapper | Yes | User code | User code | Partial | User code |
| CBF safety-filter abstraction | Yes | User code | Partial | Partial | Yes |
| PyTorch batched CBF reference | Yes | User code | No | No | No |
| Strict broadcast-shape tests | Yes | User code | No | No | No |
| Property tests for invariance | Yes | No | No | Partial | Partial |
| CPU-only reference environment | Yes | No | Yes | Yes | No |
| Parallel/vectorized toy fixture | Yes | Yes | Yes | Partial | No |
| Watchdog registry | Yes | User code | Partial | No | No |
| Failure-forensics rolling buffer | Yes | User code | No | No | No |
| Atomic checkpoint helper | Yes | User code | Partial | No | No |
| Pre-registration artifact commit | Yes | No | No | No | No |
| MLOSS-oriented packaging/docs | In progress | Yes | Yes | Yes | Partial |

## Installation

```bash
python -m pip install -e ".[test]"
```

ParallelCBF v0.1 supports Python `>=3.10,<3.14`.

## Quickstart

```python
import numpy as np

from parallelcbf.algorithms import RandomActionAlgorithm
from parallelcbf.envs import Toy2DAvoidanceEnv
from parallelcbf.ops import DefaultWatchdogRegistry, ThresholdWatchdog
from parallelcbf.safety import NaiveDistanceCBF
from parallelcbf.api import SafetyWrapper

env = SafetyWrapper(Toy2DAvoidanceEnv(), NaiveDistanceCBF())
watchdogs = DefaultWatchdogRegistry()
watchdogs.register(ThresholdWatchdog("h_hard_violation", 0.0))
algo = RandomActionAlgorithm(seed=7)

obs, _ = env.reset(seed=7)
algo.learn(env, total_timesteps=20, callback=watchdogs)
assert obs.shape == (8,)
assert not watchdogs.should_halt()
```

## Development

```bash
pytest tests
mypy --strict parallelcbf tests
pytest --cov=parallelcbf --cov-report=term-missing tests
```

The test suite pins CPU execution in `tests/conftest.py` to avoid accidental GPU
allocation in shared training environments.

## Citation

If you use ParallelCBF, please cite the repository metadata in `CITATION.cff`.

## License

Apache License 2.0. See `LICENSE`.
