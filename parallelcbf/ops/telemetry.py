"""Pydantic telemetry schemas for V24."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class V24Telemetry(BaseModel):
    """Primary V24 Phase Alpha metrics validated before dashboard emission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    stage: int = Field(ge=0, le=3)
    episode_success_rate: float = Field(ge=0.0, le=1.0)
    h_hard_violation_rate: float = Field(ge=0.0, le=1.0)
    mean_lateral_overshoot: float = Field(ge=0.0)
    mean_speed: float = Field(ge=0.0)
    policy_kl: float = Field(ge=0.0)
    critic_loss: float = Field(ge=0.0)
    actor_frozen: bool
    watchdog_halt: bool
