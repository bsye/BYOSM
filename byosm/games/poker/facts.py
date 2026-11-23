from __future__ import annotations

import random
import zlib
from collections import Counter

SIMULATIONS = 600
RANKS = "23456789TJQKA"
SUITS = "cdhs"
STRAIGHT, FLUSH = "Straight", "Flush"


def split_cards(run: str) -> list[str]:
    return [run[i : i + 2] for i in range(0, len(run), 2)]


def deck() -> list[str]:
    return [r + s for r in RANKS for s in SUITS]


def _cards(names: list[str]) -> list:
    import eval7

    return [eval7.Card(c) for c in names]


def strength(names: list[str]) -> int:
    import eval7

    return eval7.evaluate(_cards(names))


def handtype(names: list[str]) -> str:
    import eval7

    return eval7.handtype(strength(names))


def equity(hole: list[str], board: list[str], opponents: int, key: str) -> float:
    rng = random.Random(zlib.crc32(key.encode()))
    rest = [c for c in deck() if c not in hole and c not in board]
    won = 0.0
    for _ in range(SIMULATIONS):
        drawn = rng.sample(rest, 2 * opponents + 5 - len(board))
        runout = board + drawn[: 5 - len(board)]
        mine = strength(hole + runout)
        theirs = [strength(drawn[5 - len(board) + 2 * i : 7 - len(board) + 2 * i] + runout) for i in range(opponents)]
        best = max(theirs)
        won += 1.0 if mine > best else 1.0 / (1 + theirs.count(best)) if mine == best else 0.0
    return won / SIMULATIONS


def preflop_class(hole: list[str]) -> str:
    a, b = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return f"pocket {a[0]}{b[0]}"
    return f"{a[0]}{b[0]} {'suited' if a[1] == b[1] else 'offsuit'}"


def made(hole: list[str], board: list[str]) -> str:
    found = handtype(hole + board)
    return f"only the board's {found.lower()}" if len(board) == 5 and handtype(board) == found else found.lower()


def flush_draw(hole: list[str], board: list[str]) -> bool:
    suits = Counter(c[1] for c in hole + board)
    return len(board) < 5 and any(n == 4 and any(c[1] == s for c in hole) for s, n in suits.items())


def straight_outs(hole: list[str], board: list[str]) -> int:
    if len(board) == 5 or handtype(hole + board) == STRAIGHT:
        return 0
    seen = {c[0] for c in hole + board}
    off = min(SUITS, key=lambda s: sum(c[1] == s for c in hole + board))
    return sum(handtype([*hole, *board, r + off]) == STRAIGHT for r in RANKS if r not in seen)


def draws(hole: list[str], board: list[str]) -> list[str]:
    out = ["flush draw"] if flush_draw(hole, board) and handtype(hole + board) != FLUSH else []
    outs = straight_outs(hole, board)
    return out + (["open-ended straight draw"] if outs >= 2 else ["gutshot"] if outs == 1 else [])


def pot_odds(to_call: float, pot: float) -> str:
    return f"{to_call:g} to call, pot odds {to_call / (pot + to_call):.0%}"


def sentence(hole: list[str], board: list[str], opponents: int, to_call: float, pot: float) -> str:
    key = "".join(hole + board) + str(opponents)
    win = equity(hole, board, max(1, opponents), key)
    shape = preflop_class(hole) if not board else ", ".join([made(hole, board), *draws(hole, board)])
    parts = [shape, f"{win:.0%} equity against {max(1, opponents)} random hand{'s' if opponents > 1 else ''}"]
    if to_call > 0:
        parts.append(pot_odds(to_call, pot))
    return f"Facts: {'; '.join(parts)}."
