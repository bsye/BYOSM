from __future__ import annotations

from .game import CHANCE, TERMINAL, Game, Policy, State

TOLERANCE = 1e-9


def solve(game: Game) -> tuple[float, Policy]:
    if game.players != 2 or not game.zero_sum:
        raise ValueError(f"{game.name}: minimax needs a two-player zero-sum game")
    memo: dict[State, float] = {}
    policy: Policy = {}

    def worth(state: State) -> float:
        if state in memo:
            return memo[state]
        p = state.current_player()
        if p == TERMINAL:
            out = state.returns()[0]
        elif p == CHANCE:
            out = sum(pr * worth(state.apply_action(a)) for a, pr in state.chance_outcomes())
        else:
            values = [worth(state.apply_action(a)) for a in state.legal_actions()]
            best = max(values) if p == 0 else min(values)
            ties = [abs(v - best) < TOLERANCE for v in values]
            policy[state.information_state_key(p)] = [t / sum(ties) for t in ties]
            out = best
        memo[state] = out
        return out

    return worth(game.new_initial_state()), policy
