from __future__ import annotations

from collections import defaultdict

from .game import CHANCE, TERMINAL, Game, Policy, State, uniform


def probs(policy: Policy, state: State, player: int) -> list[float]:
    return policy.get(state.information_state_key(player)) or uniform(state)


def moves(policy: Policy, state: State) -> list[tuple[int, float]]:
    p = state.current_player()
    if p == CHANCE:
        return state.chance_outcomes()
    return list(zip(state.legal_actions(), probs(policy, state, p), strict=True))


def value(state: State, policy: Policy) -> list[float]:
    if state.current_player() == TERMINAL:
        return state.returns()
    total: list[float] = []
    for action, pr in moves(policy, state):
        if pr:
            got = value(state.apply_action(action), policy)
            total = [t + pr * g for t, g in zip(total, got, strict=True)] if total else [pr * g for g in got]
    return total


def _collect(state: State, player: int, policy: Policy, reach: float, depth: int, out: dict, deep: dict) -> None:
    p = state.current_player()
    if p == TERMINAL:
        return
    if p == player:
        key = state.information_state_key(player)
        out[key].append((state, reach))
        deep[key] = max(deep.get(key, 0), depth)
        for action in state.legal_actions():
            _collect(state.apply_action(action), player, policy, reach, depth + 1, out, deep)
        return
    for action, pr in moves(policy, state):
        if pr:
            _collect(state.apply_action(action), player, policy, reach * pr, depth + 1, out, deep)


def best_response(game: Game, policy: Policy, player: int) -> tuple[float, dict[str, int]]:
    root = game.new_initial_state()
    found: dict[str, list[tuple[State, float]]] = defaultdict(list)
    deep: dict[str, int] = {}
    _collect(root, player, policy, 1.0, 0, found, deep)
    chosen: dict[str, int] = {}
    memo: dict[State, float] = {}

    def worth(state: State) -> float:
        if state in memo:
            return memo[state]
        p = state.current_player()
        if p == TERMINAL:
            out = state.returns()[player]
        elif p == player:
            out = worth(state.apply_action(chosen[state.information_state_key(player)]))
        else:
            out = sum(pr * worth(state.apply_action(a)) for a, pr in moves(policy, state) if pr)
        memo[state] = out
        return out

    for key in sorted(found, key=lambda k: -deep[k]):
        states = found[key]
        legal = states[0][0].legal_actions()
        chosen[key] = max(legal, key=lambda a: sum(r * worth(s.apply_action(a)) for s, r in states))
    return worth(root), chosen


def nash_conv(game: Game, policy: Policy) -> float:
    base = value(game.new_initial_state(), policy)
    return sum(best_response(game, policy, p)[0] - base[p] for p in range(game.players))


def exploitability(game: Game, policy: Policy) -> float:
    if not (game.zero_sum and game.enumerable):
        raise ValueError(f"{game.name} is not an enumerable zero-sum game: exact exploitability does not exist")
    return nash_conv(game, policy) / game.players
