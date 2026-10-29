from __future__ import annotations

from typing import Any

from sqlalchemy import literal_column
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class JsonField(FunctionElement):
    inherit_cache = True
    name = "json_field"

    def __init__(self, column: Any, key: str):
        if not key.isidentifier():
            raise ValueError(f"not a JSON field name: {key!r}")
        super().__init__(column, literal_column(key))


def _parts(element: JsonField, compiler: Any, kw: dict[str, Any]) -> tuple[str, str]:
    column, key = element.clauses
    return compiler.process(column, **kw), key.name


@compiles(JsonField)
def _sqlite(element: JsonField, compiler: Any, **kw: Any) -> str:
    column, key = _parts(element, compiler, kw)
    return f"json_quote(json_extract({column}, '$.{key}'))"


@compiles(JsonField, "postgresql")
def _postgres(element: JsonField, compiler: Any, **kw: Any) -> str:
    column, key = _parts(element, compiler, kw)
    return f"({column} -> '{key}')"
