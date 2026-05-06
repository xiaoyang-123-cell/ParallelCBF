"""Reference algorithms for ParallelCBF."""

from parallelcbf.algorithms.causal_transformer import (
    CausalSelfAttention,
    CausalTransformer,
    CausalTransformerBC,
    CausalTransformerConfig,
    CausalTransformerHiddenState,
    TransformerBlock,
)
from parallelcbf.algorithms.causal_transformer_ppo import (
    ActorCriticOutput,
    CausalTransformerActorCritic,
    RolloutCollectionStats,
    WarmupPhase,
)
from parallelcbf.algorithms.random_action import RandomActionAlgorithm
from parallelcbf.algorithms.rollout_buffer import (
    RolloutBuffer,
    RolloutStep,
    TerminationReason,
    compute_gae_in_place,
)

__all__ = [
    "ActorCriticOutput",
    "CausalSelfAttention",
    "CausalTransformer",
    "CausalTransformerActorCritic",
    "CausalTransformerBC",
    "CausalTransformerConfig",
    "CausalTransformerHiddenState",
    "RandomActionAlgorithm",
    "RolloutBuffer",
    "RolloutCollectionStats",
    "RolloutStep",
    "TerminationReason",
    "TransformerBlock",
    "WarmupPhase",
    "compute_gae_in_place",
]
