import contextlib
import json
from types import SimpleNamespace

import byosm.deploy as deploy


def test_running_containers_are_read_from_docker_with_their_port_and_usage(monkeypatch):
    ps = {
        "Names": "byosm-q",
        "Image": "byosm-q",
        "Labels": "byosm.specialist=q,other=x",
        "Ports": "127.0.0.1:57228->8800/tcp",
        "Status": "Up 5 seconds",
    }
    stats = {"Name": "byosm-q", "CPUPerc": "1.2%", "MemUsage": "500MiB / 1GiB"}
    outputs = {"ps": json.dumps(ps), "stats": json.dumps(stats)}
    monkeypatch.setattr(deploy.shutil, "which", lambda _: "/bin/docker")
    monkeypatch.setattr(deploy, "_run", lambda *a, **k: SimpleNamespace(stdout=outputs[a[0]], returncode=0))
    [d] = deploy.deployments()
    assert (d["name"], d["port"], d["cpu"], d["memory"]) == ("q", 57228, "1.2%", "500MiB / 1GiB")
    assert "cpu" not in deploy.deployments(with_stats=False)[0]


def test_a_container_runs_read_only_with_limits_on_a_local_port(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "_run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(
        deploy.urllib.request,
        "urlopen",
        lambda *a, **k: contextlib.nullcontext(),
    )
    monkeypatch.setattr(deploy, "_free_port", lambda: 5000)
    assert deploy.deploy("Q", log=lambda _: None) == {"name": "Q", "port": 5000}
    run = calls[-1]
    assert "--read-only" in run and "127.0.0.1:5000:8800" in run and run[-1] == "byosm-q"
    assert run[run.index("--memory") + 1] == deploy.MEMORY and run[run.index("--cpus") + 1] == deploy.CPUS


def test_a_team_image_gets_the_union_of_its_members_requirements(tmp_path):
    for node, reqs in {"intent": "numpy\nonnxruntime\n", "args": "onnxruntime\ntokenizers\n"}.items():
        (tmp_path / "nodes" / node).mkdir(parents=True)
        (tmp_path / "nodes" / node / "requirements.txt").write_text(reqs)
    deploy._requirements(tmp_path)
    assert (tmp_path / "requirements.txt").read_text().split() == ["numpy", "onnxruntime", "tokenizers"]


def test_the_runtime_serves_a_team_by_name():
    from byosm.rt import System, _prep

    team = System.__new__(System)
    team.m = {"name": "assistant"}
    team.run = lambda text, action=None: {"action": "alarm_set", "text": text}
    assert team.name == "assistant" and team.predict(_prep(team, {"input": "wake me"}))["action"] == "alarm_set"
