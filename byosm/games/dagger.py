from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .best_response import exploitability, moves
from .decision import Decision, label, row
from .game import CHANCE, TERMINAL, Game, Policy, State
from .student import train

BETA = 0.5
Log = Callable[[str], None]


@dataclass(frozen=True)
class Round:
    number: int
    decisions: int
    exploitability: float


def play(game: Game, teacher: Policy, student: Policy, beta: float, rng: random.Random) -> list[tuple[State, int]]:
    seen = []
    state = game.new_initial_state()
    while state.current_player() != TERMINAL:
        p = state.current_player()
        if p != CHANCE:
            seen.append((state, p))
        actor = teacher if p == CHANCE or rng.random() < beta else student
        actions, weights = zip(*moves(actor, state), strict=True)
        state = state.apply_action(rng.choices(actions, weights=weights)[0])
    return seen


def visited(
    game: Game, teacher: Policy, student: Policy, beta: float, hands: int, rng: random.Random
) -> list[Decision]:
    counts: Counter[str] = Counter()
    states: dict[str, tuple[State, int]] = {}
    for _ in range(hands):
        for state, p in play(game, teacher, student, beta, rng):
            key = state.information_state_key(p)
            counts[key] += 1
            states[key] = (state, p)
    return [row(game, teacher, s, p, counts[k] / hands) for k, (s, p) in states.items()]


def run(game: Game, teacher: Policy, rounds: int, hands: int, seed: int, log: Log) -> list[Round]:
    rng = random.Random(seed)
    every = label(game, teacher)
    pool: dict[str, Decision] = {}
    student: Policy = {}
    out = []
    for number in range(rounds):
        for d in visited(game, teacher, student, BETA**number if number else 1.0, hands, rng):
            known = pool.get(d.key)
            pool[d.key] = Decision(d.key, d.features, d.legal, d.soft, d.weight + (known.weight if known else 0.0))
        student = train(list(pool.values()), len(game.actions), seed).policy(every)
        out.append(Round(number, len(pool), exploitability(game, student)))
        log(f"  round {number}: {len(pool)} decisions labelled, exploitability {out[-1].exploitability:.4f}")
    return out
