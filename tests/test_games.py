import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from byosm.games import registry
from byosm.games.best_response import exploitability, value
from byosm.games.cfr_exact import CFRPlus
from byosm.games.game import TERMINAL, Game

GENERIC = Path(__file__).parent.parent / "byosm" / "games"
KUHN_VALUE = -1 / 18


def kuhn_equilibrium(alpha):
    first = {"J:": [1 - alpha, alpha], "Q:": [1, 0], "K:": [1 - 3 * alpha, 3 * alpha]}
    after_bet = {"J:pb": [1, 0], "Q:pb": [2 / 3 - alpha, alpha + 1 / 3], "K:pb": [0, 1]}
    second = {"J:p": [2 / 3, 1 / 3], "J:b": [1, 0], "Q:p": [1, 0], "Q:b": [2 / 3, 1 / 3], "K:p": [0, 1], "K:b": [0, 1]}
    return {**first, **after_bet, **second}


def test_kuhn_closed_form_equilibria_are_unexploitable_and_worth_minus_one_eighteenth():
    kuhn = registry.get("kuhn")
    for alpha in (0.0, 0.2, 1 / 3):
        policy = kuhn_equilibrium(alpha)
        assert exploitability(kuhn, policy) == pytest.approx(0, abs=1e-9)
        assert value(kuhn.new_initial_state(), policy)[0] == pytest.approx(KUHN_VALUE)


def test_cfr_plus_converges_on_kuhn_and_uniform_play_is_exploitable_by_its_published_amount():
    kuhn = registry.get("kuhn")
    assert exploitability(kuhn, {}) == pytest.approx(0.458333, abs=1e-6)
    solver = CFRPlus(kuhn)
    early = exploitability(kuhn, solver.run(50))
    policy = solver.run(1950)
    assert exploitability(kuhn, policy) < min(early, 0.005)
    assert value(kuhn.new_initial_state(), policy)[0] == pytest.approx(KUHN_VALUE, abs=1e-3)


@dataclass(frozen=True)
class Throws:
    picks: tuple[int, ...] = ()

    def current_player(self):
        return TERMINAL if len(self.picks) == 2 else len(self.picks)

    def legal_actions(self):
        return [0, 1, 2]

    def chance_outcomes(self):
        return []

    def apply_action(self, action):
        return Throws((*self.picks, action))

    def information_state_key(self, player):
        return str(player)

    def returns(self):
        won = (self.picks[0] - self.picks[1]) % 3
        first = 0 if won == 0 else 1 if won == 1 else -1
        return [first, -first]


def test_a_game_registered_from_outside_the_package_is_solved_with_no_other_edit():
    game = registry.register(Game(name="throws", players=2, actions=["rock", "paper", "scissors"], start=Throws))
    assert "throws" in registry.names() and registry.get("throws") is game
    assert exploitability(game, CFRPlus(game).run(200)) < 0.05
    assert exploitability(game, {"0": [1, 0, 0], "1": [1, 0, 0]}) == pytest.approx(1.0)


def test_an_unknown_game_fails_loudly():
    with pytest.raises(ValueError, match="no rules engine registered"):
        registry.get("chess960")


def test_generic_game_modules_never_import_a_game_folder():
    folders = {p.name for p in GENERIC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    for module in GENERIC.glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                assert not set(parts) & folders, f"{module.name} imports a game folder: {node.module}"
