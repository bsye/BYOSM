import json
from types import SimpleNamespace

import pytest

from byosm.sandbox import image, supervise


class Docker:
    def __init__(self, monkeypatch, status=0, oom=False, refused=(), wait=None):
        self.calls = []
        names = ("ensure_image", "open_network", "stop", "close_network")
        for name in names:
            monkeypatch.setattr(supervise, name, lambda *a, n=name: self.calls.append(n))
        monkeypatch.setattr(supervise, "refusals", lambda job: list(refused))
        monkeypatch.setattr(supervise, "oom_killed", lambda name: oom)
        monkeypatch.setattr(supervise, "docker_cli", lambda: "docker")
        monkeypatch.setattr(supervise.signal, "signal", lambda *a: None)
        monkeypatch.setattr(supervise.subprocess, "Popen", lambda cmd: SimpleNamespace(wait=wait or (lambda: status)))


def run(monkeypatch, tmp_path, inner=None):
    if inner is not None:
        (tmp_path / "j.inner").write_text(str(inner))
    plan = {"job": "j", "container": "byosm-job-j", "image": "i", "context": "c", "torch": "t", "network": []}
    plan |= {"inner": str(tmp_path / "j.inner"), "run": ["img"], "memory_gb": 16.0}
    monkeypatch.setattr(supervise.sys, "argv", ["-c", json.dumps(plan)])
    with pytest.raises(SystemExit) as stopped:
        supervise.main()
    return stopped.value.code


def test_the_supervisor_reports_the_jobs_own_exit_code_and_cleans_up(monkeypatch, tmp_path, capsys):
    docker = Docker(monkeypatch, refused=["refused example.com:443"])
    assert run(monkeypatch, tmp_path, inner=3) == 3
    assert docker.calls == ["ensure_image", "open_network", "stop", "close_network"]
    assert "refused example.com:443; add the host to jobs.network" in capsys.readouterr().out


def test_a_container_stopped_for_memory_fails_and_says_so(monkeypatch, tmp_path, capsys):
    Docker(monkeypatch, status=137, oom=True)
    assert run(monkeypatch, tmp_path) == 137
    assert supervise.OOM_LINE.format(16.0) in capsys.readouterr().out


def test_a_cancelled_container_job_is_still_stopped_and_cleaned_up(monkeypatch, tmp_path):
    def cancelled():
        raise SystemExit(-15)

    docker = Docker(monkeypatch, wait=cancelled)
    assert run(monkeypatch, tmp_path) == -15 and docker.calls[-2:] == ["stop", "close_network"]


def test_the_training_image_is_built_only_when_missing(monkeypatch):
    built = []
    monkeypatch.setattr(image, "build_image", lambda *a: built.append(a))
    monkeypatch.setattr(image, "docker", lambda *a: SimpleNamespace(returncode=0))
    image.ensure_image("byosm-train:x", "/ctx", "idx")
    monkeypatch.setattr(image, "docker", lambda *a: SimpleNamespace(returncode=1))
    image.ensure_image("byosm-train:x", "/ctx", "idx")
    [(dockerfile, context, tag, _, args)] = built
    assert (dockerfile.name, str(context), tag, args) == (
        "train.Dockerfile",
        "/ctx",
        "byosm-train:x",
        {"TORCH_INDEX": "idx"},
    )


def test_the_training_image_is_tagged_by_the_code_snapshot_and_its_flavour(monkeypatch, tmp_path):
    import byosm.app.snapshot as snapshot

    code = tmp_path / "code" / "abc123"
    code.mkdir(parents=True)
    monkeypatch.setattr(snapshot, "frozen", lambda: code)
    cpu, gpu = image.training_image(False), image.training_image(True)
    assert cpu["image"].startswith("byosm-train:abc123-") and cpu["image"] != gpu["image"]
    assert (cpu["torch"], gpu["torch"]) == (image.TORCH_INDEX["cpu"], image.TORCH_INDEX["gpu"])
    assert (code / "train.Dockerfile").exists() and (code / "pyproject.toml").exists() and cpu["context"] == str(code)
