from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .teacher import run, strategy

SEATS = 6
FIELDS = ("caller", "random")
REPORT = "poker-table:"
VERSUS = ":vs:"
WINRATE = "winrate:"
UNIT = "mbb/hand"
INTERVAL = "(95% CI ±"
SEARCH_MS = 100
Log = Callable[[str], None]


def parsed(said: list[str]) -> dict[str, float]:
    found = [line for line in said if WINRATE in line]
    if not found:
        raise RuntimeError("the poker teacher printed no winrate")
    mean, _, rest = found[-1].partition(WINRATE)[2].partition(UNIT)
    ci, _, rest = rest.partition(INTERVAL)[2].partition(")")
    hands = rest.partition("over ")[2].split()[0]
    return {"mbb_per_hand": float(mean), "ci95": float(ci), "hands": int(hands)}


def key(name: str, field: str, search: bool) -> str:
    return f"{REPORT}{name}:{field}:{'search' if search else 'blueprint'}"


def play(ws: Path, hands_dir: Path, name: str, field: str, hands: int, search: bool, log: Log) -> dict[str, float]:
    from ...db.workspace import save_workspace_report

    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}; the table can be filled with: {', '.join(FIELDS)}")
    searching = ["--search", "--search-ms", str(SEARCH_MS)] if search else []
    args = ["eval", "--blueprint", strategy(ws, name), "--hands", str(hands), "--baseline", field]
    said = run(ws, hands_dir, [*args, "--players", str(SEATS), "--aivat", *searching], log)
    result = {**parsed(said), "seats": SEATS, "strategy": name, "field": field, "search": search}
    save_workspace_report(ws, key(name, field, search), result)
    return result


def versus(ws: Path, hands_dir: Path, focal: str, field: str, hands: int, log: Log) -> dict[str, float]:
    from ...db.workspace import save_workspace_report

    args = ["crossplay", "--focal", strategy(ws, focal), "--field", strategy(ws, field), "--hands", str(hands)]
    result = {**parsed(run(ws, hands_dir, args, log)), "seats": SEATS, "strategy": focal, "field": field}
    save_workspace_report(ws, f"{REPORT}{focal}{VERSUS}{field}", result)
    return result
