from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ..phh import BET_RAISE, CHECK_CALL, DEAL_BOARD, DEAL_HOLE, FOLD, Table, seat_of, steps

FOLDS, CHECKS, CALLS, RAISES = "fold", "check", "call", "raise"
KINDS = (FOLDS, CHECKS, CALLS, RAISES)
RAISING = {"bet", RAISES, "all_in"}
STREETS = ("preflop", "flop", "turn", "river")
HIDDEN = "????"
SHOWN = "sm"
VARIANT = "NT"
CHIPS_PER_BIG_BLIND = 100
FEWEST, MOST = 3, 6
CARD = 2
OPTIONS = {"sample": False, "search": False}
LOGGED = ("variant", "antes", "blinds_or_straddles", "min_bet", "starting_stacks", "players")


@dataclass(frozen=True)
class Asked:
    key: str
    player: str
    kind: str
    street: str
    request: dict[str, Any]
    logged: str = ""
    showdown: bool = False


def kind_of(move: str) -> str:
    return RAISES if move in RAISING else move


def cards(packed: str) -> list[str]:
    return [packed[i : i + CARD] for i in range(0, len(packed), CARD)]


def refused(hand: dict[str, Any]) -> str | None:
    blinds, seats = hand.get("blinds_or_straddles", []), len(hand.get("players", []))
    if hand.get("variant") != VARIANT:
        return "not no-limit hold'em"
    if not FEWEST <= seats <= MOST:
        return f"not {FEWEST} to {MOST} players"
    if any(hand.get("antes", [])):
        return "antes"
    if any(math.isinf(stack) for stack in hand.get("starting_stacks", [])):
        return "stacks are not logged"
    if len(blinds) != seats or not blinds[0] or not blinds[1] or any(blinds[2:]):
        return "blinds are not a small and a big blind"
    return None


def holes(steps_: list[list[str]]) -> dict[int, str]:
    found: dict[int, str] = {}
    for step in steps_:
        dealt = step[0] == "d" and step[1] == DEAL_HOLE
        shown = len(step) == 3 and step[1] == SHOWN
        if (dealt or shown) and step[-1] != HIDDEN:
            found[seat_of(step[2] if dealt else step[0])] = step[-1]
    return found


def logged_until(hand: dict[str, Any], index: int, seat: int, shown: str) -> str:
    dealt = f"d {DEAL_HOLE} p{seat + 1} "
    actions = [f"{dealt}{shown}" if a.startswith(dealt) else a for a in hand["actions"][:index]]
    kept = {key: hand[key] for key in LOGGED}
    return "".join(f"{key} = {value!r}\n" for key, value in {**kept, "actions": actions}.items())


def moved(table: Table, seat: int, move: str, to: int) -> dict[str, Any]:
    if move == FOLD:
        return {"seat": seat, "type": FOLDS}
    if move == CHECK_CALL:
        return {"seat": seat, "type": CALLS if table.owed(seat) else CHECKS}
    return {"seat": seat, "type": RAISES if table.owed_to else "bet", "to": to}


def asked(hand_id: str, hand: dict[str, Any]) -> list[Asked]:
    scale = CHIPS_PER_BIG_BLIND / hand["blinds_or_straddles"][1]
    chips = [round(x * scale) for x in hand["starting_stacks"]]
    blinds = [round(x * scale) for x in hand["blinds_or_straddles"]]
    table, played = Table(chips, blinds), steps(hand["actions"])
    known, history, found = holes(played), [], []
    setup = {
        "small_blind": blinds[0],
        "big_blind": blinds[1],
        "button_seat": len(chips) - 1,
        "players": [{"seat": seat, "stack_start": stack} for seat, stack in enumerate(chips)],
    }
    for index, step in enumerate(played):
        if step[0] == "d":
            if step[1] == DEAL_BOARD:
                table.deal(step[2])
            continue
        seat, move = seat_of(step[0]), step[1]
        if move not in (FOLD, CHECK_CALL, BET_RAISE):
            continue
        to = round(float(step[2]) * scale) if move == BET_RAISE else 0
        entry = moved(table, seat, move, to)
        if seat in known:
            board = [card for dealt in table.board for card in cards(dealt)]
            request = {
                **setup,
                "hand_id": f"{hand_id}:{index}",
                "hero_seat": seat,
                "hole_cards": cards(known[seat]),
                "board": board,
                "history": list(history),
                "options": OPTIONS,
            }
            street = STREETS[len(table.board)]
            text = logged_until(hand, index, seat, known[seat])
            player = hand["players"][seat]
            found.append(Asked(request["hand_id"], player, kind_of(entry["type"]), street, request, text))
        history.append(entry)
        table.act(seat, move, to)
    stayed = set(range(len(chips))) - table.folded
    return [replace(a, showdown=len(stayed) > 1 and a.request["hero_seat"] in stayed) for a in found]


def judged(logged: Asked, response: dict[str, Any]) -> dict[str, Any]:
    weight = sum(p["p"] for p in response.get("probs", []) if kind_of(p["type"]) == logged.kind)
    return {
        "same": kind_of(response["action"]["type"]) == logged.kind,
        "weight": weight,
        "fell_back": response.get("fallback_reason") is not None,
    }


def pooled(found: list[dict[str, Any]]) -> dict[str, Any]:
    if not found:
        return {"decisions": 0}
    count = len(found)
    return {
        "decisions": count,
        "same_kind": sum(f["same"] for f in found) / count,
        "mean_weight": sum(f["weight"] for f in found) / count,
        "fell_back": sum(f["fell_back"] for f in found) / count,
    }
