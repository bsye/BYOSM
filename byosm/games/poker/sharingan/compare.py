from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..phh import parse
from ..sealed import Log, streamed
from . import BUCKETS, blueprint, work_dir
from . import asker as asker_script
from . import engine as rival
from .arena import blueprint_mounts, ensure_image, sealed_run
from .decisions import Asked, asked, judged, pooled, refused

REPORT = "poker-decisions:"
WORK = "work"
MOUNTED_WORK = "/work"
ASKER = "/asker/asker.py"
ALL = "all"
STREET = "street:"
PLURIBUS = "Pluribus"
PROFESSIONALS = "professionals"
STAKE = "stake:"
SITE = "site:"
FULL_STACK = (8_000, 12_000)
Group = Callable[[Asked], list[str]]
Kept = Callable[[Asked], bool]
Hands = Iterable[tuple[str, dict[str, Any]]]


def pluribus_hands(limit: int | None) -> Hands:
    from ....benchmarks.pluribus import hands

    for path in hands()[:limit]:
        yield f"{path.parent.name}/{path.stem}", parse(path.read_text())


def against_pluribus(logged: Asked) -> list[str]:
    return [PLURIBUS if logged.player == PLURIBUS else PROFESSIONALS]


def by_stake(logged: Asked) -> list[str]:
    site, _, stake = logged.key.partition("/")[0].partition("-")
    return [f"{STAKE}{int(stake):05d}", f"{SITE}{site}"]


def full_stack(logged: Asked) -> bool:
    stack = logged.request["players"][logged.request["hero_seat"]]["stack_start"]
    return FULL_STACK[0] <= stack <= FULL_STACK[1]


def everyone(_: Asked) -> bool:
    return True


def shown_down(logged: Asked) -> bool:
    return logged.showdown


def collected(hands: Hands, kept: Kept) -> tuple[list[Asked], dict[str, int], Counter]:
    found, counts, skipped = [], Counter(), Counter()
    for hand_id, hand in hands:
        counts["hands"] += 1
        why = refused(hand)
        if why:
            skipped[why] += 1
            continue
        logged = asked(hand_id, hand)
        counts["hands_showing_cards"] += bool(logged)
        counts["decisions_with_cards"] += len(logged)
        found += [a for a in logged if kept(a)]
    return found, dict(counts), skipped


def answers(ws: Path, name: str, source: str, found: list[Asked], log: Log) -> list[dict[str, Any]]:
    held = blueprint(name)
    work = work_dir(ws) / WORK
    work.mkdir(exist_ok=True)
    sent, got = f"{source}.requests.jsonl", f"{source}.answers.jsonl"
    (work / sent).write_text("".join(json.dumps({"key": a.key, "request": a.request}) + "\n" for a in found))
    files = ["--requests", f"{MOUNTED_WORK}/{sent}", "--answers", f"{MOUNTED_WORK}/{got}"]
    loaded = ["--blueprint", f"{rival.MOUNTED_WEIGHTS}/{held.file}", "--buckets", f"{rival.MOUNTED_DATA}/{BUCKETS}"]
    args = ["python3", "-I", ASKER, *files, *loaded, "--sizing", held.sizing]
    mounts = {**blueprint_mounts(ws), work: MOUNTED_WORK, Path(asker_script.__file__): ASKER}
    ensure_image(log)
    command = sealed_run(mounts, args, writable=(MOUNTED_WORK,))
    streamed(command, rival.turn(ws), "the decision comparison", [name, source], log)
    return [json.loads(line) for line in (work / got).read_text().splitlines() if line]


def compared(
    ws: Path, name: str, source: str, hands: Hands, group: Group, log: Log, kept: Kept = everyone
) -> dict[str, Any]:
    from ....db.workspace import save_workspace_report

    rival.stored(ws, blueprint(name))
    found, counts, skipped = collected(hands, kept)
    log(f"asking {name} for its move at {len(found):,} logged decisions of {counts.get('hands', 0):,} hands")
    by_key = {a.key: a for a in found}
    groups: dict[str, list] = defaultdict(list)
    errors: Counter = Counter()
    for item in answers(ws, name, source, found, log):
        logged = by_key[item["key"]]
        if not item["ok"]:
            errors[item["error"]["code"]] += 1
            continue
        verdict = judged(logged, item["response"])
        for label in (ALL, *group(logged), f"{STREET}{logged.street}"):
            groups[label].append(verdict)
    made = {
        "blueprint": name,
        "source": source,
        "hands": counts.get("hands", 0),
        "hands_showing_cards": counts.get("hands_showing_cards", 0),
        "decisions_with_cards": counts.get("decisions_with_cards", 0),
        "asked": len(found),
        "skipped": dict(skipped),
        "errors": dict(errors),
        "groups": {label: pooled(verdicts) for label, verdicts in groups.items()},
    }
    save_workspace_report(ws, f"{REPORT}{name}:{source}", made)
    return made
