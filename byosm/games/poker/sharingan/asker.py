from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BIND = "127.0.0.1:8080"
BASE = f"http://{BIND}"
BATCH = 64
THREADS = 8
READY_SECONDS = 120
RATE = "1000000"
REPORT_EVERY = 200


def started(blueprint: str, buckets: str, sizing: str) -> subprocess.Popen:
    flags = ["--no-auth", "--no-search", "--bind", BIND, "--rate-rps", RATE, "--rate-burst", RATE]
    loaded = ["--blueprint", blueprint, "--buckets", buckets, "--sizing", sizing]
    quiet = {**os.environ, "SHARINGAN_LOG": "off"}
    return subprocess.Popen(["sharingan-serve", *loaded, *flags], env=quiet, stdout=subprocess.DEVNULL)


def ready(server: subprocess.Popen) -> None:
    for _ in range(READY_SECONDS * 4):
        if server.poll() is not None:
            raise RuntimeError(f"the inference server stopped with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE}/v1/health", timeout=2):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"the inference server did not answer in {READY_SECONDS} seconds")


def answered(batch: list[dict]) -> list[dict]:
    body = json.dumps({"requests": [item["request"] for item in batch]}).encode()
    asking = urllib.request.Request(f"{BASE}/v1/act/batch", body, {"content-type": "application/json"})
    with urllib.request.urlopen(asking, timeout=300) as said:
        results = json.load(said)["results"]
    return [{"key": item["key"], **result} for item, result in zip(batch, results, strict=True)]


def filed(requests: str, answers: str) -> None:
    asked = [json.loads(line) for line in Path(requests).read_text().splitlines() if line]
    batches = [asked[i : i + BATCH] for i in range(0, len(asked), BATCH)]
    with ThreadPoolExecutor(THREADS) as pool, Path(answers).open("w") as out:
        for done, results in enumerate(pool.map(answered, batches), start=1):
            out.writelines(json.dumps(result) + "\n" for result in results)
            if done % REPORT_EVERY == 0 or done == len(batches):
                print(f"progress: {min(done * BATCH, len(asked))} of {len(asked)} decisions", flush=True)


def conversed() -> None:
    for line in sys.stdin:
        print(json.dumps(answered([json.loads(line)])[0]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("blueprint", "buckets", "sizing"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("requests", "answers"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    server = started(args.blueprint, args.buckets, args.sizing)
    try:
        ready(server)
        if args.requests:
            filed(args.requests, args.answers)
        else:
            conversed()
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
