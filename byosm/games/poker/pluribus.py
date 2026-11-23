from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .moves import ALLIN, BET, CALL, CHECK, FOLD, RAISE
from .phh import BET_RAISE, CHECK_CALL, DEAL_BOARD, DEAL_HOLE, SEATS, Table, parse, seat_of, steps
from .spots import Spot, postflop_spot, preflop_spot

PLURIBUS = "Pluribus"
BIG_BLIND_CHIPS = 100
STREETS = ("Flop", "Turn", "River")
IP, OOP = "IP", "OOP"


def bb(chips: int) -> str:
    return f"{chips / BIG_BLIND_CHIPS:g}"


@dataclass
class Replay:
    table: Table
    hero: int
    preflop: list[str]
    postflop: list[str]
    aggressor: int | None = None
    flop_players: tuple[int, ...] = ()

    def name(self, seat: int) -> str:
        if len(self.flop_players) != 2:
            return SEATS[seat]
        return IP if seat == max(self.flop_players) else OOP

    def offered(self, seat: int) -> list[str]:
        t = self.table
        owed, left = t.owed(seat), t.left(seat)
        if owed >= left:
            return [FOLD, ALLIN]
        sized = RAISE if owed or not t.board else BET
        return [FOLD, CALL, sized, ALLIN] if owed else [CHECK, sized, ALLIN]

    def chosen(self, seat: int, move: str, to: int) -> str:
        t = self.table
        if move == BET_RAISE:
            if to - t.street[seat] >= t.left(seat):
                return ALLIN
            return RAISE if t.owed(seat) or not t.board else BET
        if move == CHECK_CALL:
            return ALLIN if t.owed(seat) >= t.left(seat) else CALL if t.owed(seat) else CHECK
        return FOLD

    def spot(self, decision: str, offered: list[str]) -> Spot:
        t = self.table
        if not t.board:
            row = {
                "prev_line": "/".join(self.preflop),
                "hero_pos": SEATS[self.hero],
                "hero_holding": t.holes[self.hero],
                "correct_decision": decision,
                "available_moves": repr(offered),
                "pot_size": bb(t.pot()),
            }
            return preflop_spot(row)
        boards = [*t.board, "", ""]
        row = {
            "preflop_action": "/".join(self.preflop),
            "board_flop": boards[0],
            "board_turn": boards[1],
            "board_river": boards[2],
            "aggressor_position": self.name(self.aggressor) if self.aggressor is not None else "nobody",
            "postflop_action": "/".join(self.postflop),
            "evaluation_at": STREETS[len(t.board) - 1],
            "available_moves": repr(offered),
            "pot_size": bb(t.pot()),
            "hero_position": self.name(self.hero),
            "holding": t.holes[self.hero],
            "correct_decision": decision,
        }
        return postflop_spot(row)

    def record(self, seat: int, move: str, to: int) -> None:
        t = self.table
        word = self.chosen(seat, move, to)
        if not t.board:
            amount = f"{bb(to)}bb" if word == RAISE else word
            self.preflop += [SEATS[seat], amount]
        else:
            sized = f"_{bb(to)}" if word in (BET, RAISE) else ""
            self.postflop.append(f"{self.name(seat)}_{word.upper()}{sized}")
        if move == BET_RAISE and not t.board:
            self.aggressor = seat


def decisions(text: str) -> Iterator[Spot]:
    hand = parse(text)
    hero = hand["players"].index(PLURIBUS)
    r = Replay(Table(list(hand["starting_stacks"]), list(hand["blinds_or_straddles"])), hero, [], [])
    for step in steps(hand["actions"]):
        if step[0] == "d" and step[1] == DEAL_HOLE:
            r.table.holes[seat_of(step[2])] = step[3]
            continue
        if step[0] == "d" and step[1] == DEAL_BOARD:
            if not r.table.board:
                r.flop_players = tuple(s for s in range(len(SEATS)) if s not in r.table.folded)
            r.table.deal(step[2])
            if len(r.table.board) > 1:
                r.postflop.append(f"dealcards/{step[2]}")
            continue
        seat, move = seat_of(step[0]), step[1]
        if move not in (BET_RAISE, CHECK_CALL, "f"):
            continue
        to = int(step[2]) if move == BET_RAISE else 0
        if seat == hero:
            offered = r.offered(seat)
            yield r.spot(r.chosen(seat, move, to), offered)
        r.record(seat, move, to)
        r.table.act(seat, move, to)
