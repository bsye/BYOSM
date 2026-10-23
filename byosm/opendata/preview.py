from __future__ import annotations

from pathlib import Path
from typing import Any

SAMPLE = 5000
SHOWN = 50
EXAMPLES = 3
ENCODINGS = ("utf-8-sig", "latin-1")
SNIFF_BYTES = 65536
DELIMITERS = ",;\t|"


def _separator(path: Path, encoding: str) -> str:
    import csv

    head = path.open(encoding=encoding, errors="replace").read(SNIFF_BYTES)
    try:
        return csv.Sniffer().sniff(head, delimiters=DELIMITERS).delimiter
    except csv.Error:
        return ","


def read(path: Path, rows: int | None = SAMPLE, columns: list[str] | None = None):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=rows, usecols=columns)
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns).head(rows)
    if suffix == ".json":
        return pd.read_json(path).head(rows)
    for encoding in ENCODINGS:
        try:
            sep = _separator(path, encoding)
            return pd.read_csv(path, nrows=rows, sep=sep, encoding=encoding, usecols=columns, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path.name}: not readable as UTF-8 or Latin-1")


def _profile(col: Any) -> dict[str, Any]:
    present = col.dropna()
    return {
        "name": str(col.name),
        "type": str(col.dtype),
        "missing": round(1 - len(present) / max(1, len(col)), 3),
        "distinct": int(present.nunique()),
        "examples": [str(v)[:60] for v in present.unique()[:EXAMPLES]],
    }


def preview(path: Path) -> dict[str, Any]:
    df = read(path)
    shown = df.head(SHOWN).astype(object).where(df.head(SHOWN).notna(), None)
    return {
        "file": path.name,
        "sampled": len(df),
        "columns": [_profile(df[c]) for c in df.columns],
        "rows": [{str(k): (None if v is None else str(v)) for k, v in r.items()} for r in shown.to_dict("records")],
    }
