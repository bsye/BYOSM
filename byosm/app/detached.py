from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import psutil

from ..config import STATE_DIR

JOBS_DIR = STATE_DIR / "jobs"
LOG_SUFFIX = ".log"
EXIT_SUFFIX = ".exit"
KILL_GRACE_S = 10
INTERRUPTED = "interrupted"
PID_CLOCK_SLACK_S = 1.0
WRAPPER = """
import os, signal, sys, traceback
path, entry, code = sys.argv.pop(1), sys.argv.pop(1), 1


def stop(*_):
    with open(path + ".tmp", "w") as out:
        out.write(str(-signal.SIGTERM))
    os.replace(path + ".tmp", path)
    os._exit(1)


signal.signal(signal.SIGTERM, stop)
try:
    exec(entry, {"__name__": "__main__"})
    code = 0
except SystemExit as stop:
    code = stop.code if isinstance(stop.code, int) else int(stop.code is not None)
except BaseException:
    traceback.print_exc()
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    with open(path + ".tmp", "w") as out:
        out.write(str(code))
    os.replace(path + ".tmp", path)
"""


def ending(code: int | None, cancelled: bool, stalled: bool) -> str:
    if cancelled:
        return "cancelled"
    if code == 0:
        return "done"
    if stalled:
        return "failed"
    return INTERRUPTED if code is None or code == -signal.SIGTERM else "failed"


def job_file(ws: Path, job_id: str, suffix: str) -> Path:
    return ws / JOBS_DIR / f"{job_id}{suffix}"


def spawn(ws: Path, job_id: str, entry: str, args: list[str], env: dict[str, str]) -> subprocess.Popen:
    log = job_file(ws, job_id, LOG_SUFFIX)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = job_file(ws, job_id, EXIT_SUFFIX)
    with log.open("ab") as out:
        return subprocess.Popen(
            [sys.executable, "-c", WRAPPER, str(done), entry, *args],
            cwd=ws,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def owned(pid: int, started: float) -> bool:
    try:
        return psutil.Process(pid).create_time() <= started + PID_CLOCK_SLACK_S
    except psutil.Error:
        return False


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    threading.Timer(KILL_GRACE_S, _force, args=(pid,)).start()


def _force(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def exit_code(ws: Path, job_id: str) -> int | None:
    path = job_file(ws, job_id, EXIT_SUFFIX)
    return int(path.read_text()) if path.exists() else None


def forget(ws: Path, job_id: str) -> None:
    for suffix in (LOG_SUFFIX, EXIT_SUFFIX):
        job_file(ws, job_id, suffix).unlink(missing_ok=True)


class Tail:
    def __init__(self, path: Path, skip: int) -> None:
        self.path, self.skip, self.offset, self.rest = path, skip, 0, b""

    def read(self, final: bool = False) -> list[str]:
        data = b""
        if self.path.exists():
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
        self.offset += len(data)
        done, _, self.rest = (self.rest + data).replace(b"\r", b"\n").rpartition(b"\n")
        if final:
            done, self.rest = done + b"\n" + self.rest, b""
        items = [line for line in done.decode(errors="replace").split("\n") if line.strip()]
        drop = min(self.skip, len(items))
        self.skip -= drop
        return items[drop:]
