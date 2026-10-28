import json
from types import SimpleNamespace

import pytest
import yaml

from byosm.app.job import Job
from byosm.app.jobs import ENTRY
from byosm.sandbox import choose, command, containers, launch
from byosm.sandbox.allow import allow_list, allowed, connections
from byosm.sandbox.launch import SUPERVISE_ENTRY

LIMITS = {"cpus": 4.0, "memory_gb": 16.0, "gpus": "0,1", "nice": 10, "cpu_hours": None}
SECRET = "hf_not_in_any_command"


def answer(code=0, out="", err=""):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


def after(run, flag):
    return run[run.index(flag) + 1]


def write_config(ws, **cfg):
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**yaml.safe_load((ws / "byosm.yaml").read_text()), **cfg}))


RUNC, GPU = {"Runtimes": {"runc": {}}}, {"Runtimes": {"nvidia": {}, "runc": {}}}
DESKTOP = {"Runtimes": {"runc": {}}, "OperatingSystem": "Docker Desktop"}


@pytest.mark.parametrize(
    ("platform", "device", "info", "provider", "url", "reason"),
    [
        ("darwin", "mps", None, "fake", None, choose.APPLE_GPU),
        ("darwin", "cpu", RUNC, "fake", None, choose.NOT_LINUX),
        ("linux", "cpu", None, "fake", None, choose.NO_DOCKER),
        ("linux", "cpu", DESKTOP, "fake", None, choose.NOT_LINUX),
        ("linux", "cuda", RUNC, "fake", None, choose.NO_NVIDIA),
        ("linux", "cpu", RUNC, "fake", "postgresql://db/byosm", choose.DATABASE_SERVER),
        ("linux", "cpu", RUNC, "claude-code", None, choose.HOST_LLM),
        ("linux", "cuda", GPU, "fake", None, None),
    ],
)
def test_a_job_leaves_the_container_only_for_a_stated_reason(
    ws, monkeypatch, platform, device, info, provider, url, reason
):
    monkeypatch.setattr(choose.sys, "platform", platform)
    monkeypatch.setattr(choose, "daemon", lambda: info)
    monkeypatch.setattr(choose.shutil, "which", lambda _: None)
    if url:
        monkeypatch.setenv("DATABASE_URL", url)
    assert choose.blocker(ws, [{"provider": provider}], device) == reason


def test_the_nvidia_hook_alone_is_enough_for_gpus(ws, monkeypatch):
    monkeypatch.setattr(choose.sys, "platform", "linux")
    monkeypatch.setattr(choose, "daemon", lambda: RUNC)
    monkeypatch.setattr(choose.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert choose.blocker(ws, [], "cuda") is None


def test_the_docker_daemon_is_asked_once_for_its_info(monkeypatch):
    monkeypatch.setattr(choose.shutil, "which", lambda _: "/bin/docker")
    monkeypatch.setattr(choose, "docker", lambda *a: answer(out=json.dumps(GPU)))
    assert choose.daemon() == GPU
    monkeypatch.setattr(choose, "docker", lambda *a: answer(1, err="Cannot connect to the Docker daemon"))
    assert choose.daemon() is None
    monkeypatch.setattr(choose.shutil, "which", lambda _: None)
    assert choose.daemon() is None


def test_the_allow_list_is_the_hub_every_llm_endpoint_and_the_configured_hosts(ws):
    local = {"provider": "openai", "base_url": "http://localhost:11434/v1", "model": "qwen"}
    write_config(ws, llm={"default": local, "judge": {"provider": "anthropic"}}, jobs={"network": ["data.gov.ie"]})
    rules = allow_list(ws, connections(ws, None))
    assert rules == ["huggingface.co", "hf.co", "localhost:11434", "api.anthropic.com:443", "data.gov.ie"]
    assert allow_list(ws, connections(ws, {"provider": "openai"}))[2] == "api.openai.com:443"


def test_a_rule_is_a_domain_with_its_subdomains_or_one_host_and_port():
    rules = ["hf.co", "localhost:11434", "::1:8080"]
    assert allowed("cdn-lfs-us-1.hf.co", 443, rules) and allowed("hf.co", 80, rules)
    assert not allowed("evilhf.co", 443, rules) and not allowed("hf.co.evil.com", 443, rules)
    assert allowed("localhost", 11434, rules) and not allowed("localhost", 22, rules)
    assert allowed("::1", 8080, rules) and not allowed("sub.localhost", 11434, rules)


def test_a_container_job_has_its_limits_its_own_network_and_no_secret_in_its_command(
    ws, monkeypatch, tmp_path, tmp_path_factory
):
    hub = tmp_path_factory.mktemp("hub")
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(command, "hf_home", lambda: tmp_path / "hf")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("BYOSM_DEVICE", "cpu")
    monkeypatch.setenv("BYOSM_APP_TOKEN", SECRET)
    (ws / ".env").write_text(f"HF_TOKEN={SECRET}\n")
    job = Job(id="abc", args=["build", "intent"])
    run = command.run_args(ws, job, ENTRY, "img", LIMITS, command.gpu_flags("0,1"), ws / "abc.inner")
    flags = [after(run, flag) for flag in ("--cpus", "--memory", "--gpus", "--network")]
    assert flags == ["4.0", "16384m", '"device=0,1"', "byosm-net-abc"]
    assert {"--read-only", "--init", "--env=HF_TOKEN", "--env=BYOSM_DEVICE", f"--volume={ws}:{ws}"} <= set(run)
    assert f"--volume={hub}:{hub}" in run and f"--volume={tmp_path / 'hf'}:{tmp_path / 'hf'}" not in run
    assert "--env=BYOSM_APP_TOKEN" not in run
    assert "--env=HTTPS_PROXY=http://byosm-egress-abc:3128" in run and SECRET not in " ".join(run)
    assert run[run.index("img") + 1 :][:3] == ["python", "-P", "-c"] and run[-2:] == ["build", "intent"]


def test_gpus_are_all_a_count_device_ids_or_none():
    assert command.gpu_flags("all") == ["--gpus", "all"] and command.gpu_flags("2") == ["--gpus", "2"]
    assert command.gpu_flags("none") == []


def test_container_mode_starts_a_supervisor_with_the_whole_plan(ws, monkeypatch, tmp_path):
    started = []
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(command, "hf_home", lambda: tmp_path / "hf")
    monkeypatch.setattr(launch, "device", lambda: "cuda")
    monkeypatch.setattr(launch, "blocker", lambda *a: None)
    monkeypatch.setattr(launch, "job_limits", lambda ws: {**LIMITS, "gpus": "all"})
    monkeypatch.setattr(launch, "training_image", lambda gpu: {"image": "byosm-train:x", "context": "/c", "torch": "t"})
    monkeypatch.setattr(launch, "spawn", lambda *a: started.append(a) or SimpleNamespace(pid=7))
    write_config(ws, jobs={"isolation": "container"})
    job = Job(id="abc", args=["report"])
    assert launch.start(ws, job, ENTRY).pid == 7
    [(_, job_id, cmd, env)] = started
    plan = json.loads(cmd[-1])
    assert job_id == "abc" and cmd[-2] == SUPERVISE_ENTRY and "BYOSM_JOB_ID" in env
    assert plan["image"] == "byosm-train:x" and after(plan["run"], "--gpus") == "all"
    assert job.isolation == {
        "mode": "container",
        "container": "byosm-job-abc",
        "cpus": 4.0,
        "memory_gb": 16.0,
        "gpus": "all",
        "network": ["huggingface.co", "hf.co"],
    }


def test_the_proxy_alone_joins_both_networks(monkeypatch):
    calls = []
    monkeypatch.setattr(containers, "docker", lambda *a: calls.append(a) or answer())
    containers.open_network("abc", "img", ["hf.co"])
    create, run, connect = calls
    assert create[:3] == ("network", "create", "--internal") and create[-1] == "byosm-net-abc"
    assert "--add-host=host.docker.internal:host-gateway" in run and run[-6:] == (
        "python",
        "-P",
        "-m",
        "byosm.sandbox.egress",
        "3128",
        "hf.co",
    )
    assert connect == ("network", "connect", "byosm-net-abc", "byosm-egress-abc")
    monkeypatch.setattr(containers, "docker", lambda *a: answer(1, err="network exists"))
    with pytest.raises(RuntimeError, match="network exists"):
        containers.open_network("abc", "img", [])


def test_container_state_comes_from_docker_inspect(monkeypatch):
    monkeypatch.setattr(containers, "docker", lambda *a: answer(out="true\n" if "OOMKilled" in a[2] else "4242\n"))
    assert containers.oom_killed("c") and containers.container_pid("c") == 4242
    monkeypatch.setattr(containers, "docker", lambda *a: answer(1, err="No such container"))
    assert not containers.oom_killed("c") and containers.container_pid("c") == 0


def test_the_stall_watch_reads_the_cpu_of_the_container_not_of_its_supervisor(monkeypatch):
    monkeypatch.setattr(launch, "container_pid", lambda name: 4242)
    monkeypatch.setattr(launch, "cpu_seconds", float)
    boxed = Job(id="a", args=[], pid=1, isolation={"mode": "container", "container": "byosm-job-a"})
    assert launch.worker_cpu(boxed)() == 4242.0 and launch.worker_cpu(Job(id="b", args=[], pid=9))() == 9.0
    assert launch.worker_cpu(Job(id="c", args=[])) is None
