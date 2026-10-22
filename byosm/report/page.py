from __future__ import annotations

from pathlib import Path

from ..project import Project, list_projects
from .seeds import SEEDS, seeds_block
from .systems import SYSTEM, reported_systems, system_block
from .tables import SUMMARY_HEAD, specialist_block, summary_row

SUMMARY = "summary"
OPEN = "<!-- report:"
CLOSE = " -->"


def reported() -> list[str]:
    return [n for n in list_projects() if Project(n).board()]


def blocks() -> dict[str, str]:
    names = reported()
    out = {SUMMARY: SUMMARY_HEAD + "\n" + "\n".join(summary_row(n) for n in names)}
    out.update({n: specialist_block(n) for n in names})
    out.update({f"{n}:{SEEDS}": t for n in names if (t := seeds_block(n))})
    out.update({f"{SYSTEM}:{n}": system_block(n) for n in reported_systems()})
    return out


def _markers(key: str) -> tuple[str, str]:
    return f"{OPEN}{key}{CLOSE}", f"<!-- /report:{key}{CLOSE}"


def keys_in(text: str) -> list[str]:
    out = []
    _, found, rest = text.partition(OPEN)
    while found:
        out.append(rest.partition(CLOSE)[0])
        _, found, rest = rest.partition(OPEN)
    return out


def unexpected(text: str, rendered: dict[str, str]) -> list[str]:
    keys = keys_in(text)
    return sorted({k for k in keys if k not in rendered or keys.count(k) > 1})


def replace_block(text: str, key: str, body: str) -> str | None:
    start, end = _markers(key)
    head, found, rest = text.partition(start)
    _, found_end, tail = rest.partition(end)
    if not found or not found_end:
        return None
    return f"{head}{start}\n{body}\n{end}{tail}"


def fill(text: str, rendered: dict[str, str]) -> tuple[str, list[str]]:
    missing = []
    for key, body in rendered.items():
        new = replace_block(text, key, body)
        if new is None:
            missing.append(key)
            start, end = _markers(key)
            new = f"{text.rstrip()}\n\n{start}\n{body}\n{end}\n"
        text = new
    return text, missing


def sync(readme: Path, check: bool) -> tuple[bool, list[str], list[str]]:
    old = readme.read_text() if readme.exists() else ""
    rendered = blocks()
    new, missing = fill(old, rendered)
    if not check and new != old:
        readme.write_text(new)
    return new == old, missing, unexpected(old, rendered)
