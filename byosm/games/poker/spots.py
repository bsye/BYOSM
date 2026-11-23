from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

from .facts import sentence, split_cards
from .lines import postflop_state, preflop_moves, preflop_state, street_moves
from .moves import BIG_BLIND, label_of

PREFLOP, FLOP, TURN, RIVER = "preflop", "flop", "turn", "river"
ACTION_WORDS = {"allin": "all in", "bet": "bet", "raise": "raise", "call": "call", "check": "check", "fold": "fold"}


@dataclass(frozen=True)
class Spot:
    text: str
    label: str
    street: str


def cards(run: str) -> str:
    return " ".join(split_cards(run))


def _preflop_move(word: str) -> str:
    return f"raise {word}" if word.endswith(BIG_BLIND) else ACTION_WORDS.get(word, word)


def preflop_line(line: str) -> str:
    return ", ".join(f"{who} {_preflop_move(word)}" for who, word in preflop_moves(line)) or "no action yet"


def streets(line: str) -> list[list[str]]:
    def said(who: str, word: str, size: float) -> str:
        return " ".join([who, ACTION_WORDS.get(word, word), *([f"{size:g}"] if size else [])])

    return [[said(*m) for m in street] for street in street_moves(line)]


def _offer(r: dict[str, str]) -> list[str]:
    return list(ast.literal_eval(r["available_moves"]))


def preflop_spot(r: dict[str, str]) -> Spot:
    offered = _offer(r)
    to_call, opponents = preflop_state(r["prev_line"], r["hero_pos"])
    facts = sentence(split_cards(r["hero_holding"]), [], opponents, to_call, float(r["pot_size"]))
    text = (
        f"Preflop, 6-max, 100 big blinds deep. You are {r['hero_pos']} holding {cards(r['hero_holding'])}. "
        f"Action: {preflop_line(r['prev_line'])}. Pot {r['pot_size']}. {facts} Moves: {', '.join(offered)}."
    )
    return Spot(text, label_of(r["correct_decision"], offered), PREFLOP)


def postflop_spot(r: dict[str, str]) -> Spot:
    offered = _offer(r)
    boards = [r["board_flop"], r["board_turn"], r["board_river"]]
    said = [f"Preflop: {preflop_line(r['preflop_action'])}."]
    for name, board, moves in zip((FLOP, TURN, RIVER), boards, streets(r["postflop_action"]), strict=False):
        said.append(f"{name.capitalize()} {cards(board)}: {', '.join(moves) or 'no action yet'}.")
    street = r["evaluation_at"].lower()
    to_call, opponents = postflop_state(r["preflop_action"], r["postflop_action"], r["hero_position"])
    board = split_cards("".join(boards[: [FLOP, TURN, RIVER].index(street) + 1]))
    facts = sentence(split_cards(r["holding"]), board, opponents, to_call, float(r["pot_size"]))
    text = (
        f"{street.capitalize()}, 6-max. You are {r['hero_position']} holding {cards(r['holding'])}, "
        f"{r['aggressor_position']} was the last aggressor. {' '.join(said)} "
        f"Pot {r['pot_size']}. {facts} Moves: {', '.join(offered)}."
    )
    return Spot(text, label_of(r["correct_decision"], offered), street)


def read_spots(path: Path, preflop: bool) -> list[Spot]:
    make = preflop_spot if preflop else postflop_spot
    with path.open(newline="", encoding="utf-8") as f:
        return [make(r) for r in csv.DictReader(f)]
