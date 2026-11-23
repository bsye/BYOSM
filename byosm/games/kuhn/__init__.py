from __future__ import annotations

from dataclasses import dataclass

from ..game import CHANCE, TERMINAL, Game

CARDS = "JQK"
PASS, BET = 0, 1
MOVES = "pb"
ANTE = 1


@dataclass(frozen=True)
class KuhnState:
    cards: tuple[int, ...] = ()
    history: str = ""

    def current_player(self) -> int:
        if len(self.cards) < 2:
            return CHANCE
        if self.history in ("pp", "bp", "bb", "pbp", "pbb"):
            return TERMINAL
        return len(self.history) % 2

    def legal_actions(self) -> list[int]:
        return [PASS, BET]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        left = [c for c in range(len(CARDS)) if c not in self.cards]
        return [(c, 1 / len(left)) for c in left]

    def apply_action(self, action: int) -> KuhnState:
        if self.current_player() == CHANCE:
            return KuhnState((*self.cards, action), self.history)
        return KuhnState(self.cards, self.history + MOVES[action])

    def information_state_key(self, player: int) -> str:
        return f"{CARDS[self.cards[player]]}:{self.history}"

    def returns(self) -> list[float]:
        h = self.history
        if h.endswith("bp"):
            folder = (len(h) - 1) % 2
            return [ANTE, -ANTE] if folder == 1 else [-ANTE, ANTE]
        stake = ANTE + (1 if "b" in h else 0)
        winner = 0 if self.cards[0] > self.cards[1] else 1
        return [stake, -stake] if winner == 0 else [-stake, stake]


LONGEST = 3


def encode(state: KuhnState, player: int) -> list[float]:
    card = [float(state.cards[player] == c) for c in range(len(CARDS))]
    moves = [float(i < len(state.history) and state.history[i] == m) for i in range(LONGEST) for m in MOVES]
    return card + moves


GAME = Game(
    name="kuhn",
    players=2,
    actions=["pass", "bet"],
    start=KuhnState,
    encoder=encode,
    about="Kuhn poker: three cards, one ante, one bet; its equilibrium value is -1/18 for the first player",
)
