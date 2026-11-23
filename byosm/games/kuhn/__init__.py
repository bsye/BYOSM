from __future__ import annotations

from dataclasses import dataclass

from ..game import CHANCE, TERMINAL, Game

RANKS = "JQKA"
PASS, BET = 0, 1
MOVES = "pb"
ANTE = 1
MOST_PLAYERS = len(RANKS) - 1


@dataclass(frozen=True)
class KuhnState:
    players: int = 2
    cards: tuple[int, ...] = ()
    history: str = ""

    def _over(self) -> bool:
        first = self.history.find(MOVES[BET])
        return self.history == MOVES[PASS] * self.players if first < 0 else len(self.history) == first + self.players

    def current_player(self) -> int:
        if len(self.cards) < self.players:
            return CHANCE
        return TERMINAL if self._over() else len(self.history) % self.players

    def legal_actions(self) -> list[int]:
        return [PASS, BET]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        left = [c for c in range(self.players + 1) if c not in self.cards]
        return [(c, 1 / len(left)) for c in left]

    def apply_action(self, action: int) -> KuhnState:
        if self.current_player() == CHANCE:
            return KuhnState(self.players, (*self.cards, action), self.history)
        return KuhnState(self.players, self.cards, self.history + MOVES[action])

    def information_state_key(self, player: int) -> str:
        return f"{RANKS[self.cards[player]]}:{self.history}"

    def returns(self) -> list[float]:
        seats = range(self.players)
        bet = [any(m == MOVES[BET] for m in self.history[p :: self.players]) for p in seats]
        contesting = [p for p in seats if bet[p]] or list(seats)
        put = [ANTE + bet[p] for p in seats]
        winner = max(contesting, key=lambda p: self.cards[p])
        return [float(sum(put) - put[p] if p == winner else -put[p]) for p in seats]


def encode(state: KuhnState, player: int) -> list[float]:
    card = [float(state.cards[player] == c) for c in range(len(RANKS))]
    longest = 2 * MOST_PLAYERS - 1
    moves = [float(i < len(state.history) and state.history[i] == m) for i in range(longest) for m in MOVES]
    return card + moves


def kuhn(name: str, players: int, about: str) -> Game:
    if not 2 <= players <= MOST_PLAYERS:
        raise ValueError(f"Kuhn poker is defined here for 2 to {MOST_PLAYERS} players, not {players}")
    return Game(
        name=name,
        players=players,
        actions=["pass", "bet"],
        start=KuhnState,
        encoder=encode,
        params={"players": players},
        about=about,
    )


GAME = kuhn(
    "kuhn", 2, "Kuhn poker: three cards, one ante, one bet; its equilibrium value is -1/18 for the first player"
)
THREE = kuhn("kuhn-3", 3, "Three-player Kuhn poker: four cards, one ante, one bet; the smallest poker with a table")
