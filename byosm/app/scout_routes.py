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

        cfg = load_config(ws)
        st = state(ws)
        return {
            "last_run": st["last_run"],
            "due": due(ws, cfg, time.time()),
            "every_days": every_days(cfg),
            "proposals": st["proposals"],
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
        from ..scout.proposals import ask, chosen, lab_fits, mark_sent, state

        found = next((p for p in state(ws)["proposals"] if p["id"] == pid), None)
        if found is None:
            raise HTTPException(404, f"no proposed technique {pid}")
        if not lab_fits(found):
            raise HTTPException(422, "the lab writes training code for text classifiers only so far")
        names = chosen(ws, found)
        if not names:
            raise HTTPException(422, "no promoted text classifier fits this technique")
        for name in names:
            runner.submit(["lab", "write", name, "--ask", ask(found)])
        return mark_sent(ws, pid, names)

    return router
