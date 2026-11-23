from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

CHANCE = -1
TERMINAL = -2
Policy = dict[str, list[float]]


class State(Protocol):
    def current_player(self) -> int: ...

    def legal_actions(self) -> list[int]: ...

    def chance_outcomes(self) -> list[tuple[int, float]]: ...

    def apply_action(self, action: int) -> State: ...

    def information_state_key(self, player: int) -> str: ...

    def returns(self) -> list[float]: ...


@dataclass(frozen=True)
class Game:
    name: str
    players: int
    actions: list[str]
    start: type
    encoder: Callable[[State, int], list[float]]
    zero_sum: bool = True
    enumerable: bool = True
    payoff_unit: str = "chips"
    about: str = ""
    params: dict = field(default_factory=dict)

    def new_initial_state(self) -> State:
        return self.start(**self.params)


def terminal(state: State) -> bool:
    return state.current_player() == TERMINAL


def uniform(state: State) -> list[float]:
    n = len(state.legal_actions())
    return [1.0 / n] * n
