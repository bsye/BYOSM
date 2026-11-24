from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .teacher import benchmark, run, strategy

SEATS = 6
ITSELF = "itself"
FIELDS = ("caller", "random", ITSELF)
REPORT = "poker-table:"
VERSUS = ":vs:"
AGREEMENT = "poker-agreement:"
STREETS = ("preflop", "flop", "turn", "river")
TOTAL = "total"
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
    if field == ITSELF and not search:
        raise ValueError(
            "against five copies of itself a strategy wins nothing; add --search to measure what search adds"
        )
    mode = "--search-gain" if field == ITSELF else "--search"
    searching = [mode, "--search-ms", str(SEARCH_MS)] if search else []
    others = [] if field == ITSELF else ["--baseline", field, "--aivat"]
    args = ["eval", "--blueprint", strategy(ws, name), "--hands", str(hands), "--players", str(SEATS)]
    said = run(ws, hands_dir, [*args, *others, *searching], log)
    result = {**parsed(said), "seats": SEATS, "strategy": name, "field": field, "search": search}
    save_workspace_report(ws, key(name, field, search), result)
    return result


def versus(ws: Path, hands_dir: Path, focal: str, field: str, hands: int, log: Log) -> dict[str, float]:
    from ...db.workspace import save_workspace_report

    args = ["crossplay", "--focal", strategy(ws, focal), "--field", strategy(ws, field), "--hands", str(hands)]
    result = {**parsed(run(ws, hands_dir, args, log)), "seats": SEATS, "strategy": focal, "field": field}
    save_workspace_report(ws, f"{REPORT}{focal}{VERSUS}{field}", result)
    return result


def share(text: str) -> float:
    return float(text.rstrip("%")) / 100


def agreed(said: list[str]) -> dict[str, dict[str, float]]:
    rows = {}
    for line in said:
        words = line.split()
        if len(words) == 5 and words[0].lower() in (*STREETS, TOTAL):
            decisions, covered, top, prob = int(words[1]), share(words[2]), share(words[3]), float(words[4])
            rows[words[0].lower()] = {"decisions": decisions, "covered": covered, "top_move": top, "mean_prob": prob}
    if TOTAL not in rows:
        raise RuntimeError("the poker teacher printed no agreement table")
    return rows


def agreement(ws: Path, hands_dir: Path, name: str, log: Log) -> dict:
    from ...db.workspace import save_workspace_report

    result = {"strategy": name, **agreed(benchmark(ws, hands_dir, name, log))}
    save_workspace_report(ws, f"{AGREEMENT}{name}", result)
    return result
