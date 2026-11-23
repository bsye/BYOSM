from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

from .moves import BIG_BLIND, label_of

PREFLOP, FLOP, TURN, RIVER = "preflop", "flop", "turn", "river"
DEAL = "dealcards"
ACTION_WORDS = {"allin": "all in", "bet": "bet", "raise": "raise", "call": "call", "check": "check", "fold": "fold"}


@dataclass(frozen=True)
class Spot:
    text: str
    label: str
    street: str


def cards(run: str) -> str:
    return " ".join(run[i : i + 2] for i in range(0, len(run), 2))


def _preflop_move(word: str) -> str:
    return f"raise {word}" if word.lower().endswith(BIG_BLIND) else ACTION_WORDS.get(word.lower(), word)


def preflop_line(line: str) -> str:
    parts = [p for p in line.split("/") if p]
    moves = [f"{parts[i]} {_preflop_move(parts[i + 1])}" for i in range(0, len(parts) - 1, 2)]
    return ", ".join(moves) or "no action yet"


def _postflop_move(token: str) -> str:
    who, _, rest = token.partition("_")
    words = [ACTION_WORDS.get(w.lower(), w.lower()) for w in rest.split("_")]
    return f"{who} {' '.join(words)}"


def streets(line: str) -> list[list[str]]:
    out: list[list[str]] = [[]]
    tokens = [t for t in line.split("/") if t]
    i = 0
    while i < len(tokens):
        if tokens[i] == DEAL:
            out.append([])
            i += 2
            continue
        out[-1].append(_postflop_move(tokens[i]))
        i += 1
    return out


def _offer(r: dict[str, str]) -> list[str]:
    return list(ast.literal_eval(r["available_moves"]))


def preflop_spot(r: dict[str, str]) -> Spot:
    offered = _offer(r)
    text = (
        f"Preflop, 6-max, 100 big blinds deep. You are {r['hero_pos']} holding {cards(r['hero_holding'])}. "
        f"Action: {preflop_line(r['prev_line'])}. Pot {r['pot_size']}. Moves: {', '.join(offered)}."
    )
    return Spot(text, label_of(r["correct_decision"], offered), PREFLOP)


def postflop_spot(r: dict[str, str]) -> Spot:
    offered = _offer(r)
    boards = [r["board_flop"], r["board_turn"], r["board_river"]]
    said = [f"Preflop: {preflop_line(r['preflop_action'])}."]
    for name, board, moves in zip((FLOP, TURN, RIVER), boards, streets(r["postflop_action"]), strict=False):
        said.append(f"{name.capitalize()} {cards(board)}: {', '.join(moves) or 'no action yet'}.")
    street = r["evaluation_at"].lower()
    text = (
        f"{street.capitalize()}, 6-max. You are {r['hero_position']} holding {cards(r['holding'])}, "
        f"{r['aggressor_position']} was the last aggressor. {' '.join(said)} "
        f"Pot {r['pot_size']}. Moves: {', '.join(offered)}."
    )
    return Spot(text, label_of(r["correct_decision"], offered), street)


def read_spots(path: Path, preflop: bool) -> list[Spot]:
    make = preflop_spot if preflop else postflop_spot
    with path.open(newline="", encoding="utf-8") as f:
        return [make(r) for r in csv.DictReader(f)]
