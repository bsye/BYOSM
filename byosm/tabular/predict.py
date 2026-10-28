from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from .derive import OTHER
from .score import EVERYONE, _plain

ALIVE, ENDED, UNSCORED = "alive", "ended", "unscored"


def _typed_columns(spec: Any) -> list[str]:
    sources = [how.get("from", col) for col, how in spec.derived.items() if col in spec.features]
    raw = [c for c in spec.features if c not in spec.derived]
    return list(dict.fromkeys([spec.start, *sources, *raw]))


def _read_columns(spec: Any) -> list[str]:
    return list(dict.fromkeys([*_typed_columns(spec), spec.event, *([spec.key] if spec.key else []), *spec.features]))


def inputs(ws: Path, name: str) -> dict[str, Any]:
    from .task import load

    spec = load(ws, name)
    return {"key": spec.key or (spec.shown or [None])[0], "columns": _typed_columns(spec)}


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


def _blank_state(df: Any, event: str) -> str:
    import pandas as pd

    from .snapshots import _dates

    ended = _dates(df, event).iloc[0]
    return ENDED if pd.notna(ended) and ended.date() < date.today() else UNSCORED


def predict(
    ws: Path, name: str, row: dict[str, Any], log: Callable[[str], None] = print, usual: Sequence[str] = ()
) -> dict[str, Any]:
    import joblib
    import pandas as pd

    from .derive import derive
    from .reasons import as_usual, effects
    from .snapshots import snapshot
    from .task import histories, load, model_path

    spec = load(ws, name)
    path = model_path(ws, name)
    if not path.exists():
        raise KeyError("run the forecast first")
    bundle = joblib.load(path)
    filled = dict.fromkeys(_read_columns(spec)) | {k: None if v == "" else v for k, v in row.items()}
    df = derive(pd.DataFrame([filled]).astype(object).where(lambda d: d.notna(), None), spec.derived, one_row=True)
    recipe = replace(spec.recipe(histories(spec, log)), per_cutoff=EVERYONE)
    now = snapshot(df, recipe, date.today().year)
    if now.empty:
        return {"state": _blank_state(df, spec.event), "row": row}
    unknown = [c for c in usual if c not in bundle["typical"]]
    if unknown:
        raise ValueError(f"not a feature of {name}: {', '.join(unknown)}")
    x = as_usual(_features(now, bundle), bundle["typical"], list(usual))
    chance = float(bundle["model"].predict_proba(x)[:, 1][0])
    return {
        "state": ALIVE,
        "row": row,
        "chance": round(chance, 4),
        "usual": list(usual),
        "effects": [_effect(e, now, bundle) for e in effects(bundle["model"], x, bundle["typical"])],
    }


def _effect(e: dict[str, Any], now: Any, bundle: dict[str, Any]) -> dict[str, Any]:
    return {**e, "value": _plain(now.iloc[0].get(e["feature"])), "typical": _plain(bundle["typical"][e["feature"]])}
