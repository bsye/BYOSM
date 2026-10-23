from __future__ import annotations

from typing import Any

REASONS = 3
MIN_DROP = 0.01


def typical(x: Any) -> dict[str, Any]:
    return {c: (x[c].mode().iloc[0] if x[c].dtype.name == "category" else x[c].median()) for c in x.columns}


def reasons(model: Any, rows: Any, usual: dict[str, Any]) -> list[list[dict[str, Any]]]:
    base = model.predict_proba(rows)[:, 1]
    drops = {}
    for col, value in usual.items():
        swapped = rows.copy()
        swapped[col] = value
        if rows[col].dtype.name == "category":
            swapped[col] = swapped[col].astype(rows[col].dtype)
        drops[col] = base - model.predict_proba(swapped)[:, 1]
    out = []
    for i in range(len(rows)):
        ranked = sorted(((col, float(d[i])) for col, d in drops.items()), key=lambda t: -t[1])
        out.append([{"feature": c, "drop": round(v, 4)} for c, v in ranked[:REASONS] if v >= MIN_DROP])
    return out
