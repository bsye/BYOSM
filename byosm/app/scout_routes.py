from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class Decision(BaseModel):
    status: Literal["accepted", "rejected"]


def scout_router(ws: Path, runner: Any, schedule: bool = True) -> APIRouter:
    from ..scout.schedule import run_schedule

    router = APIRouter()
    if schedule:
        threading.Thread(target=run_schedule, args=(ws, runner), daemon=True).start()

    @router.get("/api/scout")
    def scouted() -> dict:
        from ..config import load_config
        from ..scout.proposals import due, every_days, state
        from ..scout.routing import lab_fits

        cfg = load_config(ws)
        st = state(ws)
        return {
            "last_run": st["last_run"],
            "due": due(ws, cfg, time.time()),
            "every_days": every_days(cfg),
            "proposals": [{**p, "lab": lab_fits(p)} for p in st["proposals"]],
            "last_scan": st.get("last_scan"),
        }

    @router.post("/api/scout/{pid}")
    def decide(pid: str, body: Decision) -> dict:
        from ..scout.proposals import decide as decided

        try:
            return decided(ws, pid, body.status)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @router.post("/api/scout/{pid}/try")
    def try_it(pid: str) -> dict:
        from ..scout.proposals import state
        from ..scout.routing import ask, chosen, lab_fits, mark_sent

        found = next((p for p in state(ws)["proposals"] if p["id"] == pid), None)
        if found is None:
            raise HTTPException(404, f"no proposed technique {pid}")
        if not lab_fits(found):
            raise HTTPException(422, "the lab has no training contract for this kind yet")
        names = chosen(ws, found)
        if not names:
            raise HTTPException(422, "no promoted specialist or forecaster of this kind")
        for name in names:
            runner.submit(["lab", "write", name, "--ask", ask(found), "--paper", found["paper"]])
        return mark_sent(ws, pid, names)

    return router
