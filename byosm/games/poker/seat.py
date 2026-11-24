from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HIDDEN = "????"
BOARD_CUTS = (0, 3, 4, 5)
PASSIVE = {"fold": "f", "check": "cc", "call": "cc"}
RAISE = "raise"
ALL_IN = "all"
OBSERVED = "observe"


def dealt(board: list[str], before: int, upto: int) -> list[str]:
    return [f"d db {''.join(board[BOARD_CUTS[s - 1] : BOARD_CUTS[s]])}" for s in range(before + 1, upto + 1)]


def actions(request: dict) -> list[str]:
    seats = range(request["players"])
    found = [f"d dh p{i + 1} {''.join(request['hole']) if i == request['seat'] else HIDDEN}" for i in seats]
    street = 0
    for event in request["history"]:
        found += dealt(request["board"], street, event["street"])
        street = event["street"]
        verb = PASSIVE.get(event["kind"]) or f"cbr {event['to']}"
        found.append(f"p{event['seat'] + 1} {verb}")
    return found + dealt(request["board"], street, request["street"])


def hand(request: dict) -> str:
    seats = range(request["players"])
    stack = request["stacks"][0] + request["contrib"][0]
    blinds = [request["sb"], request["bb"], *[0] * (request["players"] - 2)]
    listed = {
        "antes": ["0" for _ in seats],
        "blinds_or_straddles": [str(b) for b in blinds],
        "starting_stacks": [str(stack) for _ in seats],
        "actions": [f"'{a}'" for a in actions(request)],
        "players": [f"'p{i + 1}'" for i in seats],
    }
    lines = ["variant = 'NT'", f"min_bet = {request['bb']}", *(f"{k} = [{', '.join(v)}]" for k, v in listed.items())]
    return "\n".join([*lines, ""])


def row(found: dict, meta: dict) -> dict[str, np.ndarray]:
    legal = np.zeros((1, meta["slots"]), bool)
    legal[0, found["offered"]] = True
    street = sum(1 for token in found["hist"] if token == meta["separator"])
    return {
        "tokens": np.array([[meta["start"], *found["hist"]]], np.int64),
        "last": np.array([len(found["hist"])], np.int64),
        "cards": np.array([street * meta["span"] + found["bucket"]], np.int64),
        "legal": legal,
    }


def chances(logits: np.ndarray, offered: list[int]) -> np.ndarray:
    kept = logits[0, offered].astype(np.float64)
    odds = np.exp(kept - kept.max())
    return odds / odds.sum()


def reply(request: dict, move: str) -> dict:
    words = move.split()
    if words[0] == RAISE:
        return {"id": request["id"], "action": RAISE, "to": int(words[-1])}
    return {"id": request["id"], "action": "allin" if words[0] == ALL_IN else words[0]}


class Seat:
    def __init__(self, engine: list[str], model: str, meta: dict, seed: int):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        self.engine = subprocess.Popen(engine, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.meta, self.seed = meta, seed

    def spot(self, text: str) -> dict:
        self.engine.stdin.write(json.dumps(text) + "\n")
        self.engine.stdin.flush()
        answered = self.engine.stdout.readline()
        if not answered:
            raise OSError("the engine closed its output")
        return json.loads(answered)

    def act(self, request: dict) -> dict:
        found = self.spot(hand(request))
        if "error" in found:
            return {"id": request["id"], "error": found["error"]}
        odds = chances(self.session.run(None, row(found, self.meta))[0], found["offered"])
        picked = np.random.default_rng([self.seed, request["id"]]).choice(len(odds), p=odds)
        return reply(request, found["moves"][picked])


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("engine", "abs", "model", "meta"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    seat = Seat([args.engine, "spots", "--abs", args.abs], args.model, meta, args.seed)
    for line in sys.stdin:
        request = json.loads(line)
        if OBSERVED not in request:
            print(json.dumps(seat.act(request)), flush=True)


if __name__ == "__main__":
    main()
