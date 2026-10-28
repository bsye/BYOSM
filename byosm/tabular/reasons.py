from __future__ import annotations

from typing import Any

REASONS = 3
MIN_DROP = 0.01


def typical(x: Any) -> dict[str, Any]:
    return {c: (x[c].mode().iloc[0] if x[c].dtype.name == "category" else x[c].median()) for c in x.columns}


def as_usual(rows: Any, usual: dict[str, Any], columns: list[str]) -> Any:
    swapped = rows.copy()
    for col in columns:
        swapped[col] = usual[col]
        if rows[col].dtype.name == "category":
            swapped[col] = swapped[col].astype(rows[col].dtype)
    return swapped


def _drops(model: Any, rows: Any, usual: dict[str, Any]) -> dict[str, Any]:
    base = model.predict_proba(rows)[:, 1]
    return {col: base - model.predict_proba(as_usual(rows, usual, [col]))[:, 1] for col in usual}


def effects(model: Any, row: Any, usual: dict[str, Any]) -> list[dict[str, Any]]:
    found = [{"feature": c, "drop": round(float(d[0]), 4)} for c, d in _drops(model, row, usual).items()]
    return sorted(found, key=lambda e: -abs(e["drop"]))


def reasons(model: Any, rows: Any, usual: dict[str, Any]) -> list[list[dict[str, Any]]]:
    drops = _drops(model, rows, usual)
    out = []
    for i in range(len(rows)):
        ranked = sorted(((col, float(d[i])) for col, d in drops.items()), key=lambda t: -t[1])
        out.append([{"feature": c, "drop": round(v, 4)} for c, v in ranked[:REASONS] if v >= MIN_DROP])
    return out
