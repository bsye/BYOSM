from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

LITERALS = {"true": "True", "false": "False"}
SEATS = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
DEAL_HOLE, DEAL_BOARD = "dh", "db"
FOLD, CHECK_CALL, BET_RAISE = "f", "cc", "cbr"


def parse(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(" = ")
        if sep:
            out[key.strip()] = ast.literal_eval(LITERALS.get(value.strip(), value.strip()))
    return out


@dataclass
class Table:
    stacks: list[int]
    blinds: list[int]
    holes: dict[int, str] = field(default_factory=dict)
    board: list[str] = field(default_factory=list)
    folded: set[int] = field(default_factory=set)
    street: list[int] = field(default_factory=list)
    put: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.street = list(self.blinds)
        self.put = list(self.blinds)

    @property
    def owed_to(self) -> int:
        return max(self.street)

    def owed(self, seat: int) -> int:
        return self.owed_to - self.street[seat]

    def left(self, seat: int) -> int:
        return self.stacks[seat] - self.put[seat]

    def pot(self) -> int:
        return sum(self.put)

    def deal(self, cards: str) -> None:
        self.board.append(cards)
        self.street = [0] * len(self.stacks)

    def act(self, seat: int, move: str, to: int = 0) -> None:
        if move == FOLD:
            self.folded.add(seat)
            return
        target = min(self.owed_to, self.street[seat] + self.left(seat)) if move == CHECK_CALL else to
        self.put[seat] += target - self.street[seat]
        self.street[seat] = target


def seat_of(token: str) -> int:
    return int(token.removeprefix("p")) - 1


def steps(actions: list[str]) -> list[list[str]]:
    return [a.split() for a in actions]
