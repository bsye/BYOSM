from __future__ import annotations

import hashlib
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

from .papers import ARXIV_ABS_URL, TIMEOUT_S

ARXIV_HTML = "https://arxiv.org/html/"
HEADINGS = ("h2", "h3", "h4")
METHOD_WORDS = ("method", "approach", "training", "model", "objective", "loss", "algorithm", "framework", "recipe")
MAX_CHARS = 12000
CACHE_DIR = "papers"


class _Sections(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: list[tuple[str, list[str]]] = [("", [])]
        self.heading: list[str] | None = None
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in HEADINGS:
            self.heading = []
        elif tag == "p":
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in HEADINGS and self.heading is not None:
            self.sections.append((" ".join("".join(self.heading).split()), []))
            self.heading = None
        elif tag == "p":
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if self.heading is not None:
            self.heading.append(data)
        elif self.depth:
            self.sections[-1][1].append(data)


def method_text(html: str) -> str:
    parser = _Sections()
    parser.feed(html)
    kept = [
        f"## {title}\n" + " ".join("".join(body).split())
        for title, body in parser.sections
        if any(w in title.lower() for w in METHOD_WORDS) and body
    ]
    return "\n\n".join(kept)[:MAX_CHARS]


def arxiv_id(url: str) -> str | None:
    tail = url.partition(ARXIV_ABS_URL.removeprefix("https://"))[2]
    return tail.strip("/") or None


def _fetch(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        return r.read().decode("utf-8", "replace")


def paper_method(url: str, log: Callable[[str], None], get: Callable[[str], str] = _fetch) -> str:
    from ..config import cache_root

    pid = arxiv_id(url)
    if not pid:
        return ""
    path = Path(cache_root()) / CACHE_DIR / f"{hashlib.sha1(pid.encode()).hexdigest()[:16]}.txt"
    if path.exists():
        return path.read_text()
    try:
        text = method_text(get(ARXIV_HTML + pid))
    except Exception as e:
        log(f"  could not read the full paper {pid}: {type(e).__name__}")
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    log(f"  read {len(text)} characters of method sections from {pid}")
    return text
