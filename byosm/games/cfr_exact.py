from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .game import CHANCE, TERMINAL, Game, Policy, State

Log = Callable[[str], None]


def matched(regrets: list[float]) -> list[float]:
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    return [r / total for r in positive] if total > 0 else [1.0 / len(regrets)] * len(regrets)


@dataclass
class CFRPlus:
    game: Game
    regrets: dict[str, list[float]] = field(default_factory=dict)
    totals: dict[str, list[float]] = field(default_factory=dict)
    pending: dict[str, list[float]] = field(default_factory=dict)
    iteration: int = 0

    def _walk(self, state: State, player: int, mine: float, theirs: float) -> float:
        p = state.current_player()
        if p == TERMINAL:
            return state.returns()[player]
        if p == CHANCE:
            return sum(
                pr * self._walk(state.apply_action(a), player, mine, theirs * pr) for a, pr in state.chance_outcomes()
            )
        legal = state.legal_actions()
        key = state.information_state_key(p)
        regret = self.regrets.setdefault(key, [0.0] * len(legal))
        strategy = matched(regret)
        if p != player:
            return sum(
                s * self._walk(state.apply_action(a), player, mine, theirs * s)
                for a, s in zip(legal, strategy, strict=True)
                if s
            )
        values = [
            self._walk(state.apply_action(a), player, mine * s, theirs) for a, s in zip(legal, strategy, strict=True)
        ]
        node = sum(s * v for s, v in zip(strategy, values, strict=True))
        gained = self.pending.setdefault(key, [0.0] * len(legal))
        self.pending[key] = [g + theirs * (v - node) for g, v in zip(gained, values, strict=True)]
        total = self.totals.setdefault(key, [0.0] * len(legal))
        self.totals[key] = [t + self.iteration * mine * s for t, s in zip(total, strategy, strict=True)]
        return node

    def run(self, iterations: int) -> Policy:
        root = self.game.new_initial_state()
        for _ in range(iterations):
            self.iteration += 1
            for player in range(self.game.players):
                self._walk(root, player, 1.0, 1.0)
                self._apply()
        return self.average()

    def _apply(self) -> None:
        for key, gained in self.pending.items():
            self.regrets[key] = [max(r + g, 0.0) for r, g in zip(self.regrets[key], gained, strict=True)]
        self.pending.clear()

    def average(self) -> Policy:
        return {k: matched(t) if sum(t) > 0 else [1.0 / len(t)] * len(t) for k, t in self.totals.items()}
