from __future__ import annotations

import os
from pathlib import Path

from ..config import ENV_FILE, env_file

KNOWN_KEYS = ("JEV_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")
OWNER_ONLY = 0o600


def _valid(name: str) -> str:
    if not name or not all(c.isupper() or c.isdigit() or c == "_" for c in name) or name[0].isdigit():
        raise ValueError("key names are upper case letters, digits and _ (like JEV_API_KEY)")
    return name


def _lines(ws: Path) -> list[str]:
    env = ws / ENV_FILE
    return env.read_text().splitlines() if env.exists() else []


def _name(line: str) -> str:
    return line.partition("=")[0].strip()


def list_keys(ws: Path) -> list[dict[str, object]]:
    stored = set(env_file(ws))
    names = list(dict.fromkeys([*KNOWN_KEYS, *sorted(stored)]))
    return [{"name": n, "set": n in stored or bool(os.environ.get(n)), "in_file": n in stored} for n in names]


def _write(ws: Path, lines: list[str]) -> None:
    env = ws / ENV_FILE
    env.write_text("\n".join(lines) + ("\n" if lines else ""))
    env.chmod(OWNER_ONLY)


def set_key(ws: Path, name: str, value: str) -> None:
    name = _valid(name)
    if not value or "\n" in value:
        raise ValueError("a key is one non-empty line")
    _write(ws, [ln for ln in _lines(ws) if _name(ln) != name] + [f"{name}={value}"])


def delete_key(ws: Path, name: str) -> None:
    _write(ws, [ln for ln in _lines(ws) if _name(ln) != _valid(name)])
