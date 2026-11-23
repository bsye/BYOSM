from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .best_response import moves, probs
from .game import TERMINAL, Game, Policy, State

FLOOR_WEIGHT = 1e-3


@dataclass(frozen=True)
class Decision:
    key: str
    features: list[float]
    legal: list[int]
    soft: list[float]
    weight: float


def reached(game: Game, policy: Policy) -> dict[str, tuple[State, int, float]]:
    found: dict[str, tuple[State, int, float]] = {}
    frontier: dict[State, float] = {game.new_initial_state(): 1.0}
    while frontier:
        following: dict[State, float] = defaultdict(float)
        for state, reach in frontier.items():
            p = state.current_player()
            if p == TERMINAL:
                continue
            if p >= 0:
                key = state.information_state_key(p)
                seen = found.get(key)
                found[key] = (state, p, reach + (seen[2] if seen else 0.0))
            for action, pr in moves(policy, state):
                following[state.apply_action(action)] += reach * pr
        frontier = following
    return found


def row(game: Game, teacher: Policy, state: State, player: int, weight: float, hard: bool = False) -> Decision:
    legal = state.legal_actions()
    given = probs(teacher, state, player)
    best = max(range(len(legal)), key=lambda i: given[i])
    chosen = [float(i == best) for i in range(len(legal))] if hard else given
    soft = [0.0] * len(game.actions)
    for a, pr in zip(legal, chosen, strict=True):
        soft[a] = pr
    key = state.information_state_key(player)
    return Decision(key, game.encoder(state, player), legal, soft, max(weight, FLOOR_WEIGHT))


def label(game: Game, teacher: Policy, hard: bool = False) -> list[Decision]:
    return [row(game, teacher, state, p, reach, hard) for state, p, reach in reached(game, teacher).values()]
