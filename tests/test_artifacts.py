from types import SimpleNamespace

import huggingface_hub
import pytest
import yaml

from byosm import artifacts
from byosm.project import Project
from byosm.spec import Label, TaskSpec


def _specialist(ws, repo):
    cfg = yaml.safe_load((ws / "byosm.yaml").read_text())
    (ws / "byosm.yaml").write_text(yaml.safe_dump({**cfg, "artifact_repo": repo}))
    p = Project("qa", ws)
    p.save_spec(TaskSpec(name="qa", description="q", modality="text", labels=[Label(name="a", definition="a")]))
    p.save_experiment("v1", meta={})
    return p


class Api:
    calls: list = []
    private = True

    def __init__(self, token=None):
        pass

    def create_repo(self, repo, repo_type, private, exist_ok):
        Api.calls.append(("create", repo, private))

    def model_info(self, repo):
        return SimpleNamespace(private=Api.private)

    def upload_folder(self, **kw):
        Api.calls.append(("upload", kw["repo_id"], kw["allow_patterns"]))


def test_push_uploads_weights_and_export_to_a_private_repo(ws, monkeypatch):
    p = _specialist(ws, "me/store")
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    Api.calls, Api.private = [], True
    artifacts.push(p, log=lambda m: None)
    assert Api.calls[0] == ("create", "me/store", True)
    assert Api.calls[1][2] == ["specialists/qa/experiments/*/*", "specialists/qa/export/**"]


def test_push_refuses_a_public_repo(ws, monkeypatch):
    p = _specialist(ws, "me/store")
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    Api.calls, Api.private = [], False
    with pytest.raises(RuntimeError):
        artifacts.push(p, log=lambda m: None)
    assert not any(c[0] == "upload" for c in Api.calls)


def test_missing_weights_are_pulled_from_the_repo(ws, monkeypatch):
    p = _specialist(ws, "me/store")
    asked = []

    def download(repo, repo_type, local_dir, allow_patterns, token):
        asked.append((repo, allow_patterns))
        (p.exp_dir("v1")).mkdir(parents=True, exist_ok=True)
        (p.exp_dir("v1") / "model.pt").write_bytes(b"w")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert artifacts.weights(p, "v1").read_bytes() == b"w"
    assert asked == [("me/store", ["specialists/qa/experiments/v1/*"])]
