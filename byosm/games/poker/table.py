from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .teacher import blueprint, run

SEATS = 6
FIELDS = ("caller", "random")
REPORT = "poker-table:"
WINRATE = "winrate:"
UNIT = "mbb/hand"
INTERVAL = "(95% CI ±"
SEARCH_MS = 100
Log = Callable[[str], None]


def parsed(line: str) -> dict[str, float]:
    mean, _, rest = line.partition(WINRATE)[2].partition(UNIT)
    ci, _, rest = rest.partition(INTERVAL)[2].partition(")")
    hands = rest.partition("over ")[2].split()[0]
    return {"mbb_per_hand": float(mean), "ci95": float(ci), "hands": int(hands)}


def key(iters: int, field: str, search: bool) -> str:
    return f"{REPORT}{iters}:{field}:{'search' if search else 'blueprint'}"


def play(ws: Path, hands_dir: Path, iters: int, field: str, hands: int, search: bool, log: Log) -> dict[str, float]:
    from ...db.workspace import save_workspace_report

    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}; the table can be filled with: {', '.join(FIELDS)}")
    searching = ["--search", "--search-ms", str(SEARCH_MS)] if search else []
    args = ["eval", "--blueprint", blueprint(iters), "--hands", str(hands), "--baseline", field]
    said = run(ws, hands_dir, [*args, "--players", str(SEATS), "--aivat", *searching], log)
    found = [line for line in said if WINRATE in line]
    if not found:
        raise RuntimeError("the poker teacher printed no winrate")
    result = {**parsed(found[-1]), "seats": SEATS, "field": field, "search": search, "iters": iters}
    save_workspace_report(ws, key(iters, field, search), result)
    return result
