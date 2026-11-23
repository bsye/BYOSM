from __future__ import annotations

from collections import defaultdict

FOLD, CHECK, CALL, ALLIN, BET, RAISE = "fold", "check", "call", "allin", "bet", "raise"
SIZED = (BET, RAISE)
SMALL, BIG = "small", "big"
BIG_BLIND = "bb"

DEFINITIONS = {
    FOLD: "give up the hand",
    CHECK: "pass without betting, when nothing is owed",
    CALL: "match the bet owed",
    ALLIN: "put in every remaining chip",
    f"{BET} {SMALL}": "bet the smaller of the offered bet sizes, or the only one",
    f"{BET} {BIG}": "bet the larger of two offered bet sizes",
    f"{RAISE} {SMALL}": "raise to the smaller of the offered raise sizes, or the only one",
    f"{RAISE} {BIG}": "raise to the larger of two offered raise sizes",
}
LABELS = list(DEFINITIONS)


def family(label: str) -> str:
    return label.partition(" ")[0]


def kind(move: str) -> str:
    word = move.split()[0].lower()
    return RAISE if word.endswith(BIG_BLIND) else word


def amount(move: str) -> float:
    last = move.split()[-1].lower()
    return float(last.removesuffix(BIG_BLIND))


def labelled(offered: list[str]) -> dict[str, str]:
    sized: dict[str, list[str]] = defaultdict(list)
    out: dict[str, str] = {}
    for m in offered:
        k = kind(m)
        if k in SIZED:
            sized[k].append(m)
        elif k in DEFINITIONS:
            out[m] = k
        else:
            raise ValueError(f"unknown poker move {m!r}")
    for k, moves in sized.items():
        if len(moves) > 2:
            raise ValueError(f"{len(moves)} {k} sizes offered in {offered}; the labels hold two")
        ranked = sorted(moves, key=amount) if len(moves) > 1 else moves
        out |= {m: f"{k} {SMALL if i == 0 else BIG}" for i, m in enumerate(ranked)}
    return out


def label_of(decision: str, offered: list[str]) -> str:
    found = labelled(offered)
    if decision not in found:
        raise ValueError(f"decision {decision!r} is not among the offered moves {offered}")
    return found[decision]
