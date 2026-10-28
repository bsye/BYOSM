from __future__ import annotations

from ..deploy import docker
from .egress import HOST_ALIAS, REFUSED

JOB_PREFIX = "byosm-job-"
EGRESS_PREFIX = "byosm-egress-"
NETWORK_PREFIX = "byosm-net-"
LABEL = "byosm.job"
EGRESS_MODULE = "byosm.sandbox.egress"
EGRESS_PORT = 3128
EGRESS_LIMITS = ("--cpus", "1", "--memory", "256m", "--user", "65534:65534")
SCRATCH = "/tmp"
SCRATCH_MOUNT = f"{SCRATCH}:rw,exec,nosuid,nodev"
HARDENED = ("--read-only", "--tmpfs", SCRATCH_MOUNT, "--cap-drop", "ALL", "--security-opt", "no-new-privileges")
STOP_GRACE_S = 5
ERROR_TAIL = 300


def proxy_url(job_id: str) -> str:
    return f"http://{EGRESS_PREFIX}{job_id}:{EGRESS_PORT}"


def _must(*args: str) -> None:
    out = docker(*args)
    if out.returncode:
        raise RuntimeError(f"docker {' '.join(args[:2])} failed: {out.stderr.strip()[-ERROR_TAIL:]}")


def open_network(job_id: str, image: str, rules: list[str]) -> None:
    net, proxy, label = NETWORK_PREFIX + job_id, EGRESS_PREFIX + job_id, f"{LABEL}={job_id}"
    _must("network", "create", "--internal", "--label", label, net)
    gateway = f"--add-host={HOST_ALIAS}:host-gateway"
    serve = ("python", "-P", "-m", EGRESS_MODULE, str(EGRESS_PORT), *rules)
    _must("run", "--detach", "--name", proxy, "--label", label, gateway, *HARDENED, *EGRESS_LIMITS, image, *serve)
    _must("network", "connect", net, proxy)


def refusals(job_id: str) -> list[str]:
    out = docker("logs", EGRESS_PREFIX + job_id)
    return sorted({line for line in (out.stdout + out.stderr).splitlines() if line.startswith(REFUSED)})


def close_network(job_id: str) -> None:
    docker("rm", "--force", JOB_PREFIX + job_id, EGRESS_PREFIX + job_id)
    docker("network", "rm", NETWORK_PREFIX + job_id)


def stop(name: str) -> None:
    docker("stop", "--time", str(STOP_GRACE_S), name)


def _state(name: str, field: str) -> str:
    out = docker("inspect", "--format", f"{{{{.State.{field}}}}}", name)
    return out.stdout.strip() if out.returncode == 0 else ""


def oom_killed(name: str) -> bool:
    return _state(name, "OOMKilled") == "true"


def container_pid(name: str) -> int:
    pid = _state(name, "Pid")
    return int(pid) if pid.isdigit() else 0
