from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..phh import parse
from ..sealed import Log
from . import BUCKETS, blueprint
from . import asker as asker_script
from . import engine as rival
from .arena import blueprint_mounts, ensure_image, sealed_run
from .compare import ASKER
from .decisions import CHIPS_PER_BIG_BLIND, RAISES, kind_of, pending, refused, shown_seats

STAY_OPEN = ("-i",)
NOT_YOUR_TURN = "not_hero_turn"
HAND = "hand"


def named(move: dict[str, Any], legal: dict[str, Any], unit: float) -> str:
    if kind_of(move["type"]) != RAISES:
        added = (legal.get("call") or {}).get("added")
        return f"call {added * unit:g}" if move["type"] == "call" and added else move["type"]
    most = (legal.get("raise") or {}).get("max_to")
    amount = f"{move['to'] * unit:g}"
    return f"all in for {amount}" if move.get("to") == most else f"raise to {amount}"


def moves(response: dict[str, Any], unit: float) -> list[dict[str, Any]]:
    weights: dict[str, float] = {}
    for move in response["probs"]:
        name = named(move, response.get("legal", {}), unit)
        weights[name] = weights.get(name, 0.0) + move["p"]
    return [{"move": name, "probability": weight} for name, weight in weights.items()]


@dataclass
class BlueprintAdvisor:
    engine: Any

    @classmethod
    def start(cls, ws: Path, name: str, log: Log) -> BlueprintAdvisor:
        held = blueprint(name)
        rival.stored(ws, held)
        ensure_image(log)
        loaded = ["--blueprint", f"{rival.MOUNTED_WEIGHTS}/{held.file}", "--buckets", f"{rival.MOUNTED_DATA}/{BUCKETS}"]
        args = ["python3", "-I", ASKER, *loaded, "--sizing", held.sizing]
        mounts = {**blueprint_mounts(ws), Path(asker_script.__file__): ASKER}
        piped = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "text": True}
        return cls(subprocess.Popen(sealed_run(mounts, args, flags=STAY_OPEN), **piped))

    def asked(self, request: dict[str, Any]) -> dict[str, Any]:
        self.engine.stdin.write(json.dumps({"key": HAND, "request": request}) + "\n")
        self.engine.stdin.flush()
        answered = self.engine.stdout.readline()
        if not answered:
            raise OSError("the blueprint's container closed its output")
        return json.loads(answered)

    def advise(self, hand_text: str) -> dict[str, Any]:
        hand = parse(hand_text)
        why = refused(hand)
        if why:
            raise ValueError(f"the blueprint does not play this hand: {why}")
        seats = shown_seats(hand)
        if not seats:
            raise ValueError("the history shows nobody's cards: deal the player to act their two cards")
        seat = seats[0]
        found = self.asked(pending(HAND, hand, seat).request)
        to_act = found.get("error", {}).get("details", {}).get("to_act_seat")
        if not found["ok"] and found["error"]["code"] == NOT_YOUR_TURN and to_act in seats:
            seat = to_act
            found = self.asked(pending(HAND, hand, seat).request)
        if not found["ok"]:
            raise ValueError(found["error"].get("message") or found["error"]["code"])
        answer = found["response"]
        unit = hand["blinds_or_straddles"][1] / CHIPS_PER_BIG_BLIND
        found = {"seat": seat + 1, "player": hand["players"][seat], "street": answer["street"]}
        return {**found, "moves": moves(answer, unit)}

    def close(self) -> None:
        self.engine.stdin.close()
        self.engine.wait()
