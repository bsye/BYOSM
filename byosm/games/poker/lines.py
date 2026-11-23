from __future__ import annotations

from .moves import ALLIN, BIG_BLIND, CALL, FOLD

DEAL = "dealcards"
ACTING = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
STACK_BB = 100.0
BLINDS = {"SB": 0.5, "BB": 1.0}
HEADS_UP = ("IP", "OOP")


def preflop_moves(line: str) -> list[tuple[str, str]]:
    parts = [p for p in line.split("/") if p]
    return [(parts[i], parts[i + 1].lower()) for i in range(0, len(parts) - 1, 2)]


def preflop_put(line: str) -> dict[str, float]:
    put = dict(BLINDS)
    for who, word in preflop_moves(line):
        if word.endswith(BIG_BLIND):
            put[who] = float(word.removesuffix(BIG_BLIND))
        elif word == CALL:
            put[who] = max(put.values())
        elif word == ALLIN:
            put[who] = STACK_BB
    return put


def preflop_folded(line: str) -> set[str]:
    return {who for who, word in preflop_moves(line) if word == FOLD}


def preflop_in(line: str, hero: str) -> set[str]:
    named = {who for who, _ in preflop_moves(line)}
    behind = set() if hero in named else set(ACTING[ACTING.index(hero) + 1 :]) - named
    return (named | behind | {hero}) - preflop_folded(line)


def preflop_state(line: str, hero: str) -> tuple[float, int]:
    put = preflop_put(line)
    return max(put.values()) - put.get(hero, 0.0), len(preflop_in(line, hero)) - 1


def street_moves(line: str) -> list[list[tuple[str, str, float]]]:
    out: list[list[tuple[str, str, float]]] = [[]]
    tokens = [t for t in line.split("/") if t]
    i = 0
    while i < len(tokens):
        if tokens[i] == DEAL:
            out.append([])
            i += 2
            continue
        who, _, rest = tokens[i].partition("_")
        word, _, size = rest.lower().partition("_")
        out[-1].append((who, word, float(size) if size else 0.0))
        i += 1
    return out


def postflop_state(preflop: str, postflop: str, hero: str) -> tuple[float, int]:
    moves = street_moves(postflop)
    put: dict[str, float] = {}
    for who, word, size in moves[-1]:
        put[who] = size if size else max(put.values(), default=0.0) if word == CALL else put.get(who, 0.0)
    to_call = max(put.values(), default=0.0) - put.get(hero, 0.0)
    if hero in HEADS_UP:
        return to_call, 1
    seated = {who for who, _ in preflop_moves(preflop)} | {hero}
    folded = preflop_folded(preflop) | {who for street in moves for who, word, _ in street if word == FOLD}
    return to_call, len(seated - folded) - 1
