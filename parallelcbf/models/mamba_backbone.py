"""Draft Mamba/S6 policy backbone interfaces.

This module intentionally avoids importing third-party Mamba kernels. It defines
the framework-facing structure so a concrete SSM implementation can be plugged
in later without changing algorithm code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


TensorT = TypeVar("TensorT")


@dataclass(frozen=True, slots=True)
class MambaBackboneConfig:
    """Configuration for a future Mamba/S6 sequence backbone."""

    input_dim: int
    hidden_dim: int = 128
    state_dim: int = 16
    num_layers: int = 2
    expand: int = 2
    dropout: float = 0.0
    chunk_size: int = 32


class SequenceBackbone(ABC, Generic[TensorT]):
    """Simulator-agnostic recurrent/sequence model contract."""

    @abstractmethod
    def initial_state(self, batch_size: int) -> TensorT:
        """Return an initial recurrent/SSM state."""

    @abstractmethod
    def forward_sequence(self, inputs: TensorT, state: TensorT) -> tuple[TensorT, TensorT]:
        """Process a full sequence and return outputs plus final state."""

    @abstractmethod
    def forward_chunked(self, inputs: TensorT, state: TensorT, *, chunk_size: int) -> tuple[TensorT, TensorT]:
        """Process a sequence in temporal chunks while carrying state."""


class MambaPolicyBackbone(SequenceBackbone[TensorT], ABC):
    """Abstract policy backbone for swapping GRU with Mamba/S6 blocks."""

    def __init__(self, config: MambaBackboneConfig) -> None:
        self.config = config

    @abstractmethod
    def encode_observation(self, observations: TensorT) -> TensorT:
        """Project observations into the Mamba token space."""

    @abstractmethod
    def decode_action_features(self, features: TensorT) -> TensorT:
        """Project Mamba features into policy-head features."""
