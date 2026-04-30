from typing import Any, Generic, TypeVar

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class Env(Generic[ObsT, ActT]):
    observation_space: Any
    action_space: Any
    metadata: dict[str, Any]
    reward_range: tuple[float, float]
    spec: Any

    def reset(
        self,
        *,
        seed: int | None = ...,
        options: dict[str, Any] | None = ...,
    ) -> tuple[ObsT, dict[str, Any]]: ...

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class _Box:
    low: Any
    high: Any

    def __init__(
        self,
        low: Any,
        high: Any,
        shape: tuple[int, ...] | None = ...,
        dtype: Any = ...,
    ) -> None: ...


class _Spaces:
    Box: type[_Box]


spaces: _Spaces
