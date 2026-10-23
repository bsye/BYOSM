from __future__ import annotations

from pathlib import Path
from typing import Any

from .preview import SAMPLE, SHOWN, _profile, read

KEY_HINTS = ("id", "num", "number", "code", "no", "ref", "key")
MIN_DISTINCT_SHARE = 0.5
MIN_MATCH = 0.01
CANDIDATES = 5


def _norm(name: str) -> str:
    return "".join(c for c in name.lower().lstrip("﻿") if c.isalnum())


def _keyish(df: Any, col: str) -> bool:
    values = df[col].dropna()
    return len(values) > 0 and values.nunique() / len(values) >= MIN_DISTINCT_SHARE


def _values(df: Any, col: str) -> set[str]:
    return {str(v).strip().lower() for v in df[col].dropna()}


def _pairs(left: Any, right: Any) -> list[tuple[str, str]]:
    same = [(lc, rc) for lc in left.columns for rc in right.columns if _norm(lc) == _norm(rc)]
    hinted = [
        (lc, rc)
        for lc in left.columns
        for rc in right.columns
        if (lc, rc) not in same and any(h in _norm(lc) for h in KEY_HINTS) and any(h in _norm(rc) for h in KEY_HINTS)
    ]
    return [(lc, rc) for lc, rc in same + hinted if _keyish(left, lc)]


def suggest(left_path: Path, right_path: Path) -> list[dict[str, Any]]:
    left, right = read(left_path), read(right_path)
    out = []
    for lc, rc in _pairs(left, right):
        theirs = _values(read(right_path, rows=None, columns=[rc]), rc)
        mine = _values(left, lc)
        share = len(mine & theirs) / max(1, len(mine))
        if share >= MIN_MATCH:
            out.append({"left": lc, "right": rc, "match": round(share, 4)})
    return sorted(out, key=lambda c: -c["match"])[:CANDIDATES]


def joined(left_path: Path, right_path: Path, left_key: str, right_key: str) -> dict[str, Any]:
    left = read(left_path)
    right = read(right_path, rows=None)
    lk, rk = left[left_key].astype(str).str.strip().str.lower(), right[right_key].astype(str).str.strip().str.lower()
    right = right.assign(_key=rk).drop_duplicates("_key")
    out = left.assign(_key=lk).merge(right, on="_key", how="left", suffixes=("", " (right)"), indicator=True)
    matched = float((out["_merge"] == "both").mean())
    out = out.drop(columns=["_key", "_merge"])
    shown = out.head(SHOWN).astype(object).where(out.head(SHOWN).notna(), None)
    return {
        "match": round(matched, 4),
        "sampled": min(len(left), SAMPLE),
        "right_rows": len(right),
        "columns": [_profile(out[c]) for c in out.columns],
        "rows": [{str(k): (None if v is None else str(v)) for k, v in r.items()} for r in shown.to_dict("records")],
    }
