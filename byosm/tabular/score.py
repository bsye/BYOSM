from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np

from .reasons import reasons, typical
from .snapshots import CUTOFF, LABEL, Recipe, snapshot

TOP = 100
BANDS = 10
EVERYONE = 10**9


def today(
    df: Any, r: Recipe, categorical: list[str], dropped: list[str], shown: list[str], log: Callable[[str], None]
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import pandas as pd

    from .fit import _xy, models

    year = date.today().year
    shift = year - max(r.cutoffs["test"])
    trained_on = [y + shift for y in r.cutoffs["train"]]
    train = pd.concat([snapshot(df, r, y).drop(columns=dropped) for y in trained_on], ignore_index=True)
    now = snapshot(df, replace(r, per_cutoff=EVERYONE), year).drop(columns=dropped)
    model = models(categorical)["gradient boosting"]
    x, y = _xy(train, categorical)
    model.fit(x, y)
    xnow = _xy(now, categorical)[0]
    p = model.predict_proba(xnow)[:, 1]
    log(f"  scored {len(now):,} alive on 1 Jan {year} with a model trained on {trained_on}")
    order = np.argsort(-p)[:TOP]
    features = [c for c in now.columns if c not in (LABEL, CUTOFF)]
    rows = df.loc[now.index[order], shown].astype(object).where(df.loc[now.index[order], shown].notna(), None)
    why = reasons(model, xnow.iloc[order], typical(x))
    top = [
        {
            "p": round(float(p[i]), 4),
            "shown": row,
            "features": {c: _plain(now.iloc[i][c]) for c in features},
            "why": [{**w, "value": _plain(now.iloc[i][w["feature"]])} for w in because],
        }
        for i, row, because in zip(order, rows.to_dict("records"), why, strict=True)
    ]
    edges = np.linspace(0, 1, BANDS + 1)
    counts = np.histogram(p, bins=edges)[0]
    everyone = df.loc[now.index, shown].assign(chance=p.round(4)).sort_values("chance", ascending=False)
    return (
        {
            "year": year,
            "trained_on": trained_on,
            "scored": len(now),
            "expected": round(float(p.sum())),
            "bands": [
                {"from": float(a), "to": float(b), "n": int(n)}
                for a, b, n in zip(edges, edges[1:], counts, strict=False)
            ],
            "top": top,
        },
        everyone,
        {
            "model": model,
            "year": year,
            "categorical": categorical,
            "dropped": dropped,
            "columns": list(x.columns),
            "categories": {c: list(x[c].cat.categories) for c in categorical if c in x},
            "typical": typical(x),
        },
    )


def _plain(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else str(v)
