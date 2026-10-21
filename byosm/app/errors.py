from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def missing(fn: Any, *a: Any) -> Any:
    try:
        return fn(*a)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
