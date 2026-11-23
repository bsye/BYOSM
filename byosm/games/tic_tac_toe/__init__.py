from __future__ import annotations

from dataclasses import dataclass

from ..game import TERMINAL, Game

EMPTY = -1
SIDE = 3
CELLS = SIDE * SIDE
MARKS = "xo"
LINES = (
    *((r * SIDE, r * SIDE + 1, r * SIDE + 2) for r in range(SIDE)),
    *((c, c + SIDE, c + 2 * SIDE) for c in range(SIDE)),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class TicTacToeState:
    board: tuple[int, ...] = (EMPTY,) * CELLS

    def winner(self) -> int | None:
        for a, b, c in LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def current_player(self) -> int:
        if self.winner() is not None or EMPTY not in self.board:
            return TERMINAL
        return sum(cell != EMPTY for cell in self.board) % 2

    def legal_actions(self) -> list[int]:
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return []

    def apply_action(self, action: int) -> TicTacToeState:
        return TicTacToeState((*self.board[:action], self.current_player(), *self.board[action + 1 :]))

    def information_state_key(self, player: int) -> str:
        return "".join("." if cell == EMPTY else MARKS[cell] for cell in self.board)

    def returns(self) -> list[float]:
        won = self.winner()
        if won is None:
            return [0.0, 0.0]
        return [1.0, -1.0] if won == 0 else [-1.0, 1.0]


GAME = Game(
    name="tic-tac-toe",
    players=2,
    actions=[f"cell {i + 1}" for i in range(CELLS)],
    start=TicTacToeState,
    about="Tic-tac-toe: perfect information, solved exactly by minimax; perfect play draws",
)
