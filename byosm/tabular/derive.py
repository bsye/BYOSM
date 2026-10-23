from __future__ import annotations

from typing import Any

MIN_COUNT = 200
NUMERIC_SHARE = 0.95
MAX_CATEGORIES = 250
OTHER = "other"


def _words(col: Any) -> Any:
    return col.fillna("").astype(str).str.split()


def _part(col: Any, sep: str, index: int) -> Any:
    return col.fillna("").astype(str).str.split(sep).str[index].str.strip().str.upper().replace("", None)


def _first(col: Any, n: int) -> Any:
    return col.str[:n]


def _keywords(col: Any, groups: dict[str, str], other: str) -> Any:
    def pick(v: Any) -> str:
        text = str(v).upper()
        return next((g for word, g in groups.items() if word.upper() in text), other)

    return col.map(pick)


def _frequent(col: Any, min_count: int = MIN_COUNT) -> Any:
    counts = col.map(col.value_counts())
    return col.where(counts >= min_count)


STEPS = {
    "part": lambda col, a: _part(col, a.get("sep", ","), a.get("index", -1)),
    "first": lambda col, a: _first(col, a["n"]),
    "word_count": lambda col, a: _words(col).str.len(),
    "contains": lambda col, a: (
        col.fillna("").astype(str).str.upper().str.contains(a["word"].upper(), regex=False).astype(int)
    ),
    "keywords": lambda col, a: _keywords(col, a["groups"], a.get("other", "other")),
    "frequent": lambda col, a: _frequent(col, a.get("min_count", MIN_COUNT)),
}


POPULATION_STEPS = ("frequent",)


def derive(df: Any, columns: dict[str, dict[str, Any]], one_row: bool = False) -> Any:
    out = df.copy()
    for name, how in columns.items():
        col = out[how.get("from", name)]
        for step in how["steps"]:
            kind, args = (step, {}) if isinstance(step, str) else next(iter(step.items()))
            if not (one_row and kind in POPULATION_STEPS):
                col = STEPS[kind](col, args or {})
        out[name] = col
    return out


def typed(df: Any, features: list[str], categorical: list[str]) -> tuple[Any, list[str]]:
    import pandas as pd

    out, cats = df.copy(), list(categorical)
    for col in (c for c in features if c not in categorical):
        present = out[col].dropna()
        numbers = pd.to_numeric(present, errors="coerce")
        if len(present) and numbers.notna().mean() >= NUMERIC_SHARE:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            cats.append(col)
    for col in cats:
        top = out[col].value_counts().index[:MAX_CATEGORIES]
        out[col] = out[col].where(out[col].isin(top) | out[col].isna(), OTHER)
    return out, cats
