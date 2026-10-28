from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from xml.etree import ElementTree

ARXIV = "https://export.arxiv.org/api/query"
HF_DAILY = "https://huggingface.co/api/daily_papers"
ATOM = "{http://www.w3.org/2005/Atom}"
PER_QUERY = 15
HF_LIMIT = 50
TIMEOUT_S = 30
SUMMARY_CHARS = 700
DAY_CHARS = 10
DAILY = "daily"
ARXIV_ABS_URL = "https://arxiv.org/abs/"
AREAS = {
    "text": 'abs:"text classification" AND (abs:"small" OR abs:"few-shot" OR abs:"encoder") AND cat:cs.CL',
    "tools": '(abs:"function calling" OR abs:"tool selection" OR abs:"tool use") AND cat:cs.CL',
    "series": 'abs:"time series forecasting" AND (abs:"foundation" OR abs:"pretrain") AND cat:cs.LG',
    "turns": '(abs:"end-of-turn" OR abs:"turn-taking" OR abs:"endpointing")',
}


@dataclass(frozen=True)
class Paper:
    url: str
    title: str
    summary: str
    area: str
    published: str = ""


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        return r.read()


def _clean(text: str) -> str:
    return " ".join(text.split())[:SUMMARY_CHARS]


def arxiv(area: str, query: str, get: Callable[[str], bytes] = _get) -> list[Paper]:
    params = {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": PER_QUERY}
    root = ElementTree.fromstring(get(f"{ARXIV}?{urllib.parse.urlencode(params)}"))
    out = []
    for e in root.iter(f"{ATOM}entry"):
        url = (e.findtext(f"{ATOM}id") or "").strip().replace("http://", "https://")
        title, summary = _clean(e.findtext(f"{ATOM}title") or ""), _clean(e.findtext(f"{ATOM}summary") or "")
        out.append(Paper(url, title, summary, area, (e.findtext(f"{ATOM}published") or "")[:DAY_CHARS]))
    return out


def daily(get: Callable[[str], bytes] = _get) -> list[Paper]:
    rows = json.loads(get(f"{HF_DAILY}?limit={HF_LIMIT}"))
    papers = [r.get("paper") or {} for r in rows]
    return [
        Paper(ARXIV_ABS_URL + p["id"], _clean(p.get("title", "")), _clean(p.get("summary", "")), DAILY, _day(p))
        for p in papers
        if p.get("id")
    ]


def _day(paper: dict) -> str:
    return str(paper.get("publishedAt") or "")[:DAY_CHARS]


def fetch(log: Callable[[str], None], get: Callable[[str], bytes] = _get) -> list[Paper]:
    found: list[Paper] = []
    for area, query in AREAS.items():
        try:
            found += arxiv(area, query, get)
        except Exception as e:
            log(f"  arXiv {area}: {type(e).__name__}: {str(e)[:120]}")
    try:
        found += daily(get)
    except Exception as e:
        log(f"  Hugging Face daily papers: {type(e).__name__}: {str(e)[:120]}")
    unique = {p.url: p for p in found}
    log(f"  {len(unique)} recent papers from arXiv and Hugging Face")
    return list(unique.values())
