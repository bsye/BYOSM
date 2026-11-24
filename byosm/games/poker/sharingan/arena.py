from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from ....deploy import DOCKER_DIR, build_image, docker, docker_cli
from .. import seat, teacher
from ..distill import ABSTRACTION
from ..portable import exported
from ..sealed import SEALED, Log, limits, mounted, streamed
from ..table import REPORT, VERSUS
from . import BLUEPRINTS, DATA, WEIGHTS, work_dir
from . import engine as rival

DOCKERFILE = DOCKER_DIR / "poker-table.Dockerfile"
IMAGE = "byosm-poker-table"
PINNED = ("numpy", "onnxruntime")
SEAT = "/seat/seat.py"
STUDENTS = "/teacher"
ENGINE = "/usr/local/bin/poker-engine"
BENCH = "bench"
BOT = "byosm"
HOUSE = ("nit", "tag", "lag", "maniac", "station", "random")
DEALS = 20_000
SEATS = 6
TIMEOUT_MS = 20_000
MOST_FALLBACKS = 0.001
RAW = "raw"
CORRECTED = "aivat"
SAID_REPORT = f"report {BOT}:"


def pins() -> dict[str, str]:
    return {name.upper(): version(name) for name in PINNED}


def image() -> str:
    parts = [DOCKERFILE.read_text(), teacher.image(), rival.image(), json.dumps(pins(), sort_keys=True)]
    return f"{IMAGE}:{hashlib.sha256(''.join(parts).encode()).hexdigest()[:12]}"


def ensure_image(log: Log) -> None:
    if docker("image", "inspect", image()).returncode == 0:
        return
    teacher.ensure_image(log)
    rival.ensure_image(log)
    build_image(DOCKERFILE, DOCKER_DIR, image(), log, {"TEACHER": teacher.image(), "RIVAL": rival.image(), **pins()})


def seated(ws: Path, student: str) -> str:
    graph, meta = exported(ws, student)
    argv = ["python3", "-I", SEAT, "--engine", ENGINE]
    for flag, name in (("--abs", f"{student}{ABSTRACTION}"), ("--model", graph.name), ("--meta", meta.name)):
        argv += [flag, f"{STUDENTS}/{name}"]
    return f"ext:{BOT}={json.dumps(argv)}"


def entrant(ws: Path, name: str) -> tuple[str, list[str]]:
    if name in HOUSE:
        return name, []
    if name in BLUEPRINTS:
        rival.stored(ws, BLUEPRINTS[name])
        return rival.policy(BLUEPRINTS[name]), rival.loading(BLUEPRINTS[name])
    return seated(ws, name), []


def sealed_run(mounts: dict[Path, str], args: list[str], writable: tuple[str, ...] = ()) -> list[str]:
    cpus, memory = limits()
    held = ["--cpus", cpus, "--memory", memory, *mounted(mounts, writable)]
    return [docker_cli(), "run", "--rm", *SEALED, *held, image(), *args]


def blueprint_mounts(ws: Path) -> dict[Path, str]:
    return {work_dir(ws) / WEIGHTS: rival.MOUNTED_WEIGHTS, work_dir(ws) / DATA: rival.MOUNTED_DATA}


def command(ws: Path, args: list[str]) -> list[str]:
    mounts = {**blueprint_mounts(ws), teacher.work_dir(ws): STUDENTS, Path(seat.__file__): SEAT}
    return sealed_run(mounts, args)


def estimate(words: list[str], label: str) -> dict[str, float] | None:
    if label not in words:
        return None
    at = words.index(label)
    return {"mbb_per_hand": float(words[at + 1]), "ci95": float(words[at + 3])}


def parsed(said: list[str]) -> dict:
    scored = [line.split() for line in said if RAW in line.split()]
    counted = [line.split() for line in said if " deals, " in line]
    if not scored or not counted:
        raise RuntimeError("the table printed no result")
    answered = [json.loads(line.partition(SAID_REPORT)[2]) for line in said if SAID_REPORT in line]
    raw, corrected = estimate(scored[0], RAW), estimate(scored[0], CORRECTED)
    return {
        **(corrected or raw),
        RAW: raw,
        "corrected": corrected is not None,
        "deals": int(counted[-1][0]),
        "hands": int(counted[-1][2]),
        "seats": SEATS,
        "requests": sum(a["requests"] for a in answered),
        "fallbacks": sum(a["fallbacks"] for a in answered),
    }


def played(ws: Path, hero: str, field: str, deals: int, log: Log) -> dict:
    (one, loads_one), (five, loads_five) = entrant(ws, hero), entrant(ws, field)
    args = [BENCH, "match", "--a", one, "--b", five, "--players", str(SEATS), "--deals", str(deals)]
    args += [*(loads_one or loads_five), "--ext-workers", limits()[0], "--ext-timeout-ms", str(TIMEOUT_MS)]
    ensure_image(log)
    result = parsed(streamed(command(ws, args), rival.turn(ws), "the poker table", [hero, field], log))
    if result["fallbacks"] > MOST_FALLBACKS * max(result["requests"], 1):
        raise RuntimeError(
            f"our seat failed to answer {result['fallbacks']} of {result['requests']} decisions: "
            "the table would be scoring the fallback move, not the student"
        )
    return {**result, "strategy": hero, "field": field}


def versus(ws: Path, hero: str, field: str, deals: int, log: Log) -> dict:
    from ....db.workspace import save_workspace_report

    result = played(ws, hero, field, deals, log)
    save_workspace_report(ws, f"{REPORT}{hero}{VERSUS}{field}", result)
    return result
