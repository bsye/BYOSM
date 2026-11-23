from __future__ import annotations

import importlib
import pkgutil

from .game import Game

PACKAGE = "byosm.games"
ATTRIBUTE = "GAME"
_GAMES: dict[str, Game] = {}


def register(game: Game) -> Game:
    _GAMES[game.name] = game
    return game


def _discover() -> None:
    package = importlib.import_module(PACKAGE)
    for found in pkgutil.iter_modules(package.__path__):
        if found.ispkg:
            module = importlib.import_module(f"{PACKAGE}.{found.name}")
            game = getattr(module, ATTRIBUTE, None)
            if isinstance(game, Game):
                register(game)


def names() -> list[str]:
    _discover()
    return sorted(_GAMES)


def get(name: str) -> Game:
    _discover()
    if name not in _GAMES:
        raise ValueError(f"no rules engine registered for the game {name!r}; known: {', '.join(sorted(_GAMES))}")
    return _GAMES[name]
