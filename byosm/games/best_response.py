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
    memo: dict[State, list[float]] = {}

    def worth(at: State) -> list[float]:
        if at not in memo:
            if at.current_player() == TERMINAL:
                memo[at] = at.returns()
            else:
                parts = [[pr * g for g in worth(at.apply_action(a))] for a, pr in moves(policy, at) if pr]
                memo[at] = [sum(column) for column in zip(*parts, strict=True)]
        return memo[at]

    return worth(state)


def _collect(root: State, player: int, policy: Policy) -> tuple[dict, dict]:
    found: dict[str, list[tuple[State, float]]] = defaultdict(list)
    deep: dict[str, int] = {}
    frontier: dict[State, float] = {root: 1.0}
    depth = 0
    while frontier:
        following: dict[State, float] = defaultdict(float)
        for state, reach in frontier.items():
            p = state.current_player()
            if p == TERMINAL:
                continue
            if p == player:
                key = state.information_state_key(player)
                found[key].append((state, reach))
                deep[key] = depth
                for action in state.legal_actions():
                    following[state.apply_action(action)] += reach
            else:
                for action, pr in moves(policy, state):
                    if pr:
                        following[state.apply_action(action)] += reach * pr
        frontier, depth = following, depth + 1
    return found, deep


def best_response(game: Game, policy: Policy, player: int) -> tuple[float, dict[str, int]]:
    root = game.new_initial_state()
    found, deep = _collect(root, player, policy)
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
