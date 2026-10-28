import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest
import yaml

from byosm.app import jobs
from byosm.app.hub import Hub
from byosm.app.jobs import Runner
from byosm.resources import GB
from byosm.sandbox import limits, process

QUICK = "print('ok', flush=True)"
TIMEOUT_SECONDS = 20


def wait_for(check):
    end = time.time() + TIMEOUT_SECONDS
    while not check():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def set_jobs(ws, **settings):
    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "jobs": settings}))


class FakeProc:
    def __init__(self):
        self.niced, self.caps = None, {}

    def nice(self, value=None):
        if value is None:
            return 0
        self.niced = value

    def rlimit(self, which, value=None):
        if value is None:
            return (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        self.caps[which] = value


def test_jobs_run_as_processes_with_limits_taken_from_the_machine(ws, monkeypatch):
    monkeypatch.setattr(limits.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(limits.psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * GB))
    assert limits.job_limits(ws) == {"cpus": 7.0, "memory_gb": 48.0, "gpus": "all", "nice": 10, "cpu_hours": None}


def test_byosm_yaml_sets_every_limit(ws):
    set_jobs(ws, cpus=2, memory_gb=8, gpus="0,1", nice=5, cpu_hours=12)
    assert limits.job_limits(ws) == {"cpus": 2.0, "memory_gb": 8.0, "gpus": "0,1", "nice": 5, "cpu_hours": 12}


@pytest.mark.parametrize(("device", "capped"), [("cpu", True), ("cuda", False)])
def test_on_linux_a_process_job_is_niced_and_its_address_space_capped_unless_on_cuda(monkeypatch, device, capped):
    proc = FakeProc()
    monkeypatch.setattr(process.sys, "platform", "linux")
    monkeypatch.setattr(process.psutil, "Process", lambda pid: proc)
    shown = process.restrict(1, {"nice": 10, "memory_gb": 48.0, "cpu_hours": 2}, device)
    memory = (48 * GB, 48 * GB)
    assert proc.niced == 10 and proc.caps.get(resource.RLIMIT_AS) == (memory if capped else None)
    assert proc.caps[resource.RLIMIT_CPU] == (7200, 7200)
    assert shown == {"mode": "process", "nice": 10, "cpu_hours": 2, **({"memory_gb": 48.0} if capped else {})}


def test_on_a_mac_a_process_job_is_only_niced(monkeypatch):
    proc = FakeProc()
    monkeypatch.setattr(process.sys, "platform", "darwin")
    monkeypatch.setattr(process.psutil, "Process", lambda pid: proc)
    shown = process.restrict(1, {"nice": 10, "memory_gb": 48.0, "cpu_hours": 2}, "mps")
    assert shown == {"mode": "process", "nice": 10} and proc.caps == {}


def test_a_real_child_process_gets_the_nice_level():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    try:
        shown = process.restrict(child.pid, {"nice": 10, "memory_gb": 1.0, "cpu_hours": None}, "cpu")
        assert psutil.Process(child.pid).nice() == 10 and shown["nice"] == 10
    finally:
        child.wait()


def test_a_job_shows_its_isolation_and_keeps_it_after_a_restart(ws, monkeypatch):
    monkeypatch.setattr(jobs, "ENTRY", QUICK)
    runner = Runner(ws, Hub(), parallel=1)
    job = runner.submit(["report"])
    wait_for(lambda: job.status == "done")
    assert job.public()["isolation"] == job.isolation and (job.isolation["mode"], job.isolation["nice"]) == (
        "process",
        10,
    )
    assert Runner(ws, Hub(), parallel=1).jobs[job.id].public()["isolation"] == job.isolation
