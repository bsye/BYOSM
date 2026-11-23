from __future__ import annotations

from dataclasses import dataclass

from ..game import CHANCE, TERMINAL, Game

RANKS = "JQK"
DECK = 2 * len(RANKS)
FOLD, CALL, RAISE = 0, 1, 2
SYMBOLS = "fcr"
RAISE_SIZES = (2, 4)
MAX_RAISES = 2
ANTE = 1
PAIR = 10
BOARD_CARD = 3


def rank(card: int) -> int:
    return card // 2


def closed(moves: str) -> bool:
    return len(moves) >= 2 and moves[-1] == SYMBOLS[CALL]


@dataclass(frozen=True)
class LeducState:
    cards: tuple[int, ...] = ()
    rounds: tuple[str, ...] = ("",)

    def current_player(self) -> int:
        moves = self.rounds[-1]
        if len(self.cards) < 2:
            return CHANCE
        if moves.endswith(SYMBOLS[FOLD]):
            return TERMINAL
        if closed(moves):
            return CHANCE if len(self.rounds) == 1 else TERMINAL
        return len(moves) % 2

    def legal_actions(self) -> list[int]:
        moves = self.rounds[-1]
        facing = moves.endswith(SYMBOLS[RAISE])
        raising = moves.count(SYMBOLS[RAISE]) < MAX_RAISES
        return [a for a, ok in ((FOLD, facing), (CALL, True), (RAISE, raising)) if ok]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        left = [c for c in range(DECK) if c not in self.cards]
        return [(c, 1 / len(left)) for c in left]

    def apply_action(self, action: int) -> LeducState:
        if self.current_player() == CHANCE:
            dealt = (*self.cards, action)
            return LeducState(dealt, (*self.rounds, "") if len(dealt) == BOARD_CARD else self.rounds)
        return LeducState(self.cards, (*self.rounds[:-1], self.rounds[-1] + SYMBOLS[action]))

    def information_state_key(self, player: int) -> str:
        board = RANKS[rank(self.cards[2])] if len(self.cards) > 2 else ""
        return f"{RANKS[rank(self.cards[player])]}{board}:{'/'.join(self.rounds)}"

    def _put(self) -> tuple[list[int], int | None]:
        put = [ANTE, ANTE]
        for size, moves in zip(RAISE_SIZES, self.rounds, strict=False):
            for i, move in enumerate(moves):
                p = i % 2
                if move == SYMBOLS[FOLD]:
                    return put, p
                put[p] = max(put) + (size if move == SYMBOLS[RAISE] else 0)
        return put, None

    def _strength(self, player: int) -> int:
        mine = rank(self.cards[player])
        return mine + (PAIR if mine == rank(self.cards[2]) else 0)

    def returns(self) -> list[float]:
        put, folder = self._put()
        if folder is None:
            ahead = self._strength(0) - self._strength(1)
            if ahead == 0:
                return [0.0, 0.0]
            folder = 1 if ahead > 0 else 0
        won = put[folder]
        return [won, -won] if folder == 1 else [-won, won]


GAME = Game(
    name="leduc",
    players=2,
    actions=["fold", "call", "raise"],
    start=LeducState,
    about="Leduc poker: six cards, two betting rounds of 2 and 4 chips, one shared card; the classic small poker benchmark",
)
