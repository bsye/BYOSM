from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from .derive import OTHER
from .score import EVERYONE, _plain

ALIVE, ENDED = "alive", "ended"


def inputs(ws: Path, name: str) -> dict[str, Any]:
    from .task import load

    spec = load(ws, name)
    sources = [how.get("from", col) for col, how in spec.derived.items() if col in spec.features]
    raw = [c for c in spec.features if c not in spec.derived]
    return {"key": spec.key or (spec.shown or [None])[0], "columns": list(dict.fromkeys([spec.start, *sources, *raw]))}


def lookup(ws: Path, name: str, key: str, log: Callable[[str], None]) -> dict[str, Any] | None:
    from .task import load, read_table

    spec = load(ws, name)
    column = inputs(ws, name)["key"]
    df = read_table(spec.source, log, spec.file)
    hit = df[df[column].astype(str).str.strip() == key.strip()]
    return None if hit.empty else {k: _plain(v) for k, v in hit.iloc[0].items()}


def _features(df: Any, bundle: dict[str, Any]) -> Any:
    import pandas as pd

    from .fit import _xy

    x = _xy(df.drop(columns=bundle["dropped"]), [])[0]
    for col in bundle["columns"]:
        if col not in x:
            x[col] = None
    x = x[bundle["columns"]]
    for col in x.columns:
        cats = bundle["categories"].get(col)
        if cats is None:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        else:
            value = x[col].where(x[col].isin(cats) | x[col].isna(), OTHER if OTHER in cats else None)
            x[col] = pd.Categorical(value, categories=cats)
    return x


def predict(ws: Path, name: str, row: dict[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    import joblib
    import pandas as pd

    from .derive import derive
    from .reasons import reasons
    from .snapshots import snapshot
    from .task import histories, load, model_path

    spec = load(ws, name)
    path = model_path(ws, name)
    if not path.exists():
        raise KeyError("run the forecast first")
    bundle = joblib.load(path)
    df = derive(pd.DataFrame([row]).astype(object).where(lambda d: d.notna(), None), spec.derived, one_row=True)
    recipe = replace(spec.recipe(histories(spec, log)), per_cutoff=EVERYONE)
    now = snapshot(df, recipe, date.today().year)
    if now.empty:
        return {"state": ENDED, "row": row}
    x = _features(now, bundle)
    chance = float(bundle["model"].predict_proba(x)[:, 1][0])
    why = reasons(bundle["model"], x, bundle["typical"])[0]
    return {
        "state": ALIVE,
        "row": row,
        "chance": round(chance, 4),
        "why": [{**w, "value": _plain(now.iloc[0].get(w["feature"]))} for w in why],
    }
