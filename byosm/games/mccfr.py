from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cfr_exact import matched
from .game import CHANCE, TERMINAL, Game, Policy, State


def pick(rng: random.Random, actions: list[int], weights: list[float]) -> int:
    return rng.choices(actions, weights=weights)[0]


@dataclass
class ExternalSampling:
    game: Game
    seed: int = 0
    regrets: dict[str, list[float]] = field(default_factory=dict)
    totals: dict[str, list[float]] = field(default_factory=dict)
    iteration: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _strategy(self, state: State, p: int) -> tuple[str, list[int], list[float]]:
        legal = state.legal_actions()
        key = state.information_state_key(p)
        return key, legal, matched(self.regrets.setdefault(key, [0.0] * len(legal)))

    def _walk(self, state: State, player: int) -> float:
        p = state.current_player()
        if p == TERMINAL:
            return state.returns()[player]
        if p == CHANCE:
            actions, weights = zip(*state.chance_outcomes(), strict=True)
            return self._walk(state.apply_action(pick(self.rng, list(actions), list(weights))), player)
        key, legal, strategy = self._strategy(state, p)
        if p != player:
            total = self.totals.setdefault(key, [0.0] * len(legal))
            self.totals[key] = [t + s for t, s in zip(total, strategy, strict=True)]
            return self._walk(state.apply_action(pick(self.rng, legal, strategy)), player)
        values = [self._walk(state.apply_action(a), player) for a in legal]
        node = sum(s * v for s, v in zip(strategy, values, strict=True))
        self.regrets[key] = [r + v - node for r, v in zip(self.regrets[key], values, strict=True)]
        return node

    def run(self, iterations: int) -> Policy:
        root = self.game.new_initial_state()
        for _ in range(iterations):
            self.iteration += 1
            for player in range(self.game.players):
                self._walk(root, player)
        return self.average()

    def average(self) -> Policy:
        return {k: matched(t) if sum(t) > 0 else [1.0 / len(t)] * len(t) for k, t in self.totals.items()}
