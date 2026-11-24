from __future__ import annotations

from typing import Any

RANKS = "23456789TJQKA"
SUITS = "cdhs"
TEN = "10"
SYMBOLS = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}
HOLE, BOARDS = 2, (0, 3, 4, 5)
FEWEST, MOST = 2, 6
SYSTEM = "You read screenshots of a poker table. Report only what is visible in the image. Never guess a hidden card."
PROMPT = (
    "List what this poker table shows. Cards are written as rank then suit, with ranks 2-9, T, J, Q, K, A and suits "
    "c (clubs), d (diamonds), h (hearts), s (spades): for example 6c, Ad, Th. hero_cards are the two face-up cards "
    "of the player labelled as you. board is the face-up community cards in the middle, left to right, or an empty "
    "list. players lists every seat still at the table going clockwise around it, starting from the top left; for "
    "each give the name, the chip stack as a number, whether the seat still holds cards, whether it is the player "
    "labelled as you, whether it has the dealer button, and the last action shown beside it in words, or an empty "
    "string. pot is the number shown in the middle."
)
PLAYER = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "stack": {"type": "number"},
        "has_cards": {"type": "boolean"},
        "is_you": {"type": "boolean"},
        "has_dealer_button": {"type": "boolean"},
        "last_action": {"type": "string"},
    },
    "required": ["name", "stack", "has_cards", "is_you", "has_dealer_button", "last_action"],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "hero_cards": {"type": "array", "items": {"type": "string"}},
        "board": {"type": "array", "items": {"type": "string"}},
        "players": {"type": "array", "items": PLAYER},
        "pot": {"type": "number"},
    },
    "required": ["hero_cards", "board", "players", "pot"],
}


def card(written: str) -> str:
    text = written.strip().replace(TEN, "T")
    for symbol, suit in SYMBOLS.items():
        text = text.replace(symbol, suit)
    found = f"{text[:1].upper()}{text[1:2].lower()}"
    if len(text) != HOLE or found[0] not in RANKS or found[1] not in SUITS:
        raise ValueError(f"the picture was read as the card {written!r}, which is not a card")
    return found


def seated(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not FEWEST <= len(players) <= MOST:
        raise ValueError(f"the picture was read as {len(players)} players; a table has {FEWEST} to {MOST}")
    marked = {flag: [i for i, p in enumerate(players) if p[flag]] for flag in ("is_you", "has_dealer_button")}
    if len(marked["is_you"]) != 1:
        raise ValueError("the picture does not show exactly one seat as yours")
    if len(marked["has_dealer_button"]) != 1:
        raise ValueError("the picture does not show exactly one dealer button")
    first = (marked["has_dealer_button"][0] + 1) % len(players)
    return players[first:] + players[:first]


def checked(found: dict[str, Any]) -> dict[str, Any]:
    hero, board = [card(c) for c in found["hero_cards"]], [card(c) for c in found["board"]]
    if len(hero) != HOLE:
        raise ValueError(f"the picture was read as {len(hero)} cards in your hand, not {HOLE}")
    if len(board) not in BOARDS:
        raise ValueError(f"the picture was read as {len(board)} cards on the board")
    if len({*hero, *board}) != len(hero) + len(board):
        raise ValueError("the picture was read with the same card twice")
    players = seated(found["players"])
    return {
        "hero": next(i for i, p in enumerate(players) if p["is_you"]),
        "cards": hero,
        "board": board,
        "names": [p["name"] for p in players],
        "stacks": [p["stack"] for p in players],
        "in_hand": [p["has_cards"] for p in players],
        "said": [p["last_action"] for p in players],
        "pot": found["pot"],
    }


def read(image: bytes, llm: Any) -> dict[str, Any]:
    return checked(llm.json(SYSTEM, PROMPT, SCHEMA, images=[image]))
