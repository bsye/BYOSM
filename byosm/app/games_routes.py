from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ENGINE_GONE = "the engine stopped answering; ask again"


class Hand(BaseModel):
    hand: str


def games_router(ws: Path) -> APIRouter:
    router = APIRouter()
    advisors: dict[str, Any] = {}
    blueprints: dict[str, Any] = {}
    starting = threading.Lock()

    def advisor(name: str) -> Any:
        from ..games.poker.advisor import Advisor

        with starting:
            if name not in advisors:
                advisors[name] = Advisor.start(ws, None, name, lambda _: None)
            return advisors[name]

    @router.get("/api/games/{name}")
    def game(name: str) -> dict[str, Any]:
        from ..games.poker.bot import detail
        from ..games.poker.health import of_bot

        found = detail(ws, name)
        if not found:
            raise HTTPException(404, f"no game bot named {name}")
        return {**found, "health": of_bot(ws, name)}

    @router.post("/api/games/{name}/advise")
    def advise(name: str, body: Hand) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            found = advisor(name).advise(body.hand)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except OSError as e:
            advisors.pop(name, None)
            raise HTTPException(503, ENGINE_GONE) from e
        return {**found, "ms": (time.perf_counter() - started) * 1000}

    def blueprint_advisor(name: str) -> Any:
        from ..games.poker.sharingan.advice import BlueprintAdvisor

        with starting:
            if name not in blueprints:
                blueprints[name] = BlueprintAdvisor.start(ws, name, lambda _: None)
            return blueprints[name]

    @router.post("/api/blueprints/{name}/advise")
    def blueprint_advise(name: str, body: Hand) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            found = blueprint_advisor(name).advise(body.hand)
        except (ValueError, SyntaxError) as e:
            raise HTTPException(422, str(e)) from e
        except OSError as e:
            blueprints.pop(name, None)
            raise HTTPException(503, ENGINE_GONE) from e
        return {**found, "ms": (time.perf_counter() - started) * 1000}

    def close() -> None:
        for running in (*advisors.values(), *blueprints.values()):
            running.close()

    router.on_shutdown.append(close)
    return router
