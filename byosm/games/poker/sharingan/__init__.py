from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ....config import STATE_DIR

CODE = "https://github.com/markokraemer/poker-sharingan.git"
COMMIT = "6a11d9b4e44e4819ed5eacb53552118ed1374f1b"
LICENCE = "apache-2.0"
FOLDER = "poker-sharingan"
WEIGHTS = "weights"
DATA = "data"
BUCKETS = "buckets-200"


@dataclass(frozen=True)
class Blueprint:
    repo: str
    revision: str
    file: str
    parts: int
    sizing: str

    def part_names(self) -> list[str]:
        return [f"{self.file}.part-{i:02d}" for i in range(self.parts)]


DEFAULT = "sharingan-2"
BLUEPRINTS = {
    DEFAULT: Blueprint(
        "markokraemer/sharingan-2", "c97c4736260db5b34d5e6ebd0a1544fccb401cf1", "sharingan-2.plcp", 7, "medium"
    ),
}


def blueprint(name: str) -> Blueprint:
    if name not in BLUEPRINTS:
        raise ValueError(f"no published blueprint named {name!r}; known: {', '.join(BLUEPRINTS)}")
    return BLUEPRINTS[name]


def work_dir(ws: Path) -> Path:
    found = ws / STATE_DIR / FOLDER
    found.mkdir(parents=True, exist_ok=True)
    return found
