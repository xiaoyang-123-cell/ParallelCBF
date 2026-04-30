# Contributing

Thank you for helping improve ParallelCBF.

## Development Setup

```bash
python -m pip install -e ".[test]"
pytest tests
mypy --strict parallelcbf tests
```

## Pull Requests

- Keep public methods fully type hinted.
- Add tests for new behavior.
- Keep reference tests CPU-only.
- Update documentation when changing public APIs.

## Scope

ParallelCBF v0.1 focuses on simulator-agnostic contracts and CPU reference
implementations. Isaac Lab, Mamba, and PPO integrations are planned for v0.2.
