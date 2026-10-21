import json
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from byosm.cli import app
from byosm.export import export
from byosm.export.bundle import BUNDLE_FORMAT, BUNDLE_MARKER, clear_out_dir


def test_export_refuses_a_directory_that_is_not_a_bundle(ws):
    out = ws / "mine"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError) as err:
        export("anything", exp="v0", out=out, log=lambda _: None)
    assert "not a byosm bundle" in str(err.value)
    assert (out / "notes.txt").read_text() == "keep me"


def test_export_refuses_a_foreign_manifest(ws):
    out = ws / "other"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"format": "someone-else"}))
    with pytest.raises(FileExistsError):
        clear_out_dir(out)
    assert (out / "manifest.json").exists()


def test_export_refuses_a_file(ws):
    out = ws / "file.txt"
    out.write_text("x")
    with pytest.raises(FileExistsError):
        clear_out_dir(out)
    assert out.read_text() == "x"


@pytest.mark.parametrize("contents", [{}, {"manifest.json": json.dumps({"format": BUNDLE_FORMAT}), "model.onnx": ""}])
def test_export_replaces_an_empty_directory_or_a_previous_bundle(ws, contents):
    out = ws / "export"
    out.mkdir()
    for name, text in contents.items():
        (out / name).write_text(text)
    clear_out_dir(out)
    assert [f.name for f in out.iterdir()] == [BUNDLE_MARKER]


def _command_paths(group: typer.Typer, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = [(*prefix, c.name or c.callback.__name__.replace("_", "-")) for c in group.registered_commands]
    for sub in group.registered_groups:
        paths += _command_paths(sub.typer_instance, (*prefix, sub.name))
    return paths


def test_every_command_prints_its_help():
    paths = _command_paths(app)
    assert ("suite", "add") in paths
    runner = CliRunner()
    for path in [(), *paths]:
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output, result.exception)


def _bundle(out):
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"format": BUNDLE_FORMAT}))
    (out / "model.onnx").write_text("good")


def test_export_with_a_bad_experiment_keeps_the_previous_bundle(ws):
    out = ws / "export"
    _bundle(out)
    with pytest.raises(FileNotFoundError):
        export("anything", exp="does-not-exist", out=out, log=lambda _: None)
    assert (out / "model.onnx").read_text() == "good"


def test_an_interrupted_export_can_be_redone(ws, monkeypatch):
    import byosm.export as ex

    def crash(*_):
        raise KeyboardInterrupt

    def frontend(_sp, _kind, out):
        (out / "window.npy").write_text("")
        return {}, [], None, {}

    monkeypatch.setattr(ex, "Specialist", lambda *_: object())
    monkeypatch.setattr(ex.TaskSpec, "load", lambda _: None)
    monkeypatch.setattr(ex, "_calibrated_decision", lambda *_: ({}, {}))
    monkeypatch.setattr(ex, "frontend_kind", lambda _: "text")
    monkeypatch.setattr(ex, "build_frontend", frontend)
    monkeypatch.setattr(ex, "export_model", crash)
    out = ws / "export"
    with pytest.raises(KeyboardInterrupt):
        export("anything", exp="v0", out=out, log=lambda _: None)
    with pytest.raises(KeyboardInterrupt):
        export("anything", exp="v0", out=out, log=lambda _: None)
    assert not (out / "manifest.json").exists()


def _stub_export(monkeypatch, failing_step):
    import byosm.export as ex

    def fail(*_, **__):
        raise RuntimeError("export parity check failed")

    def frontend(_sp, _kind, out):
        return {}, [], None, {}

    monkeypatch.setattr(ex, "Specialist", lambda *_: SimpleNamespace(spec=None))
    monkeypatch.setattr(ex.TaskSpec, "load", lambda _: None)
    monkeypatch.setattr(ex, "_calibrated_decision", lambda *_: ({}, {}))
    monkeypatch.setattr(ex, "frontend_kind", lambda _: "text")
    monkeypatch.setattr(ex, "build_frontend", frontend)
    monkeypatch.setattr(ex, "export_model", lambda *a: (a[5] / "model.onnx").write_text(""))
    monkeypatch.setattr(ex, "QUANTIZE_MIN_BYTES", -1)
    monkeypatch.setattr(ex, "manifest_for", lambda *_: {"format": BUNDLE_FORMAT})
    monkeypatch.setattr(ex.bundle, "requirements", lambda *_: [])
    monkeypatch.setattr(ex.bundle, "readme", lambda *_: "")
    monkeypatch.setattr(ex, "verify", lambda *_: None)
    monkeypatch.setattr(ex, failing_step, fail)


@pytest.mark.parametrize("step", ["verify", "quantize"])
def test_a_bundle_that_fails_verification_is_not_left_exported(ws, monkeypatch, step):
    from byosm.export.bundle import is_exported

    _stub_export(monkeypatch, step)
    out = ws / "export"
    with pytest.raises(RuntimeError) as err:
        export("anything", exp="v0", out=out, log=lambda _: None, precision="int8" if step == "quantize" else "fp32")
    assert "parity" in str(err.value)
    assert not is_exported(out)
    assert not (out / "manifest.json").exists()
    assert (out / BUNDLE_MARKER).exists()
    clear_out_dir(out)


def test_a_verified_bundle_is_exported(ws, monkeypatch):
    from byosm.export.bundle import is_exported

    _stub_export(monkeypatch, "quantize")
    out = ws / "export"
    export("anything", exp="v0", out=out, log=lambda _: None, precision="fp32")
    assert is_exported(out)
    assert not (out / BUNDLE_MARKER).exists()


def test_private_publish_refuses_an_existing_public_repo(ws, monkeypatch, tmp_path):
    import types

    import huggingface_hub
    import pytest

    from byosm.hub import upload

    calls = []

    class Api:
        def create_repo(self, repo_id, private, exist_ok):
            calls.append("create")

        def model_info(self, repo_id):
            return types.SimpleNamespace(private=False)

        def upload_folder(self, **kw):
            calls.append("upload")

    out = ws / "specialists" / "x" / "export"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text('{"experiment": "v1"}')
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(upload, "is_exported", lambda path: True)
    monkeypatch.setattr(upload, "_stage", lambda *a: tmp_path)
    with pytest.raises(RuntimeError) as err:
        upload.publish("x", "me/x", private=True, log=lambda m: None)
    assert "already exists and is public" in str(err.value)
    assert calls == ["create"]


def test_containerize_builds_the_export_folder_and_refuses_an_unexported_specialist(ws, monkeypatch):
    import shutil
    import subprocess

    from byosm.cli.ship import MODEL_DOCKERFILE
    from byosm.spec import Label, TaskSpec

    root = ws / "specialists" / "q"
    root.mkdir(parents=True)
    TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")]).save(
        root / "spec.yaml"
    )
    assert "export first" in CliRunner().invoke(app, ["containerize", "q"]).output
    (root / "export").mkdir()
    (root / "export" / "manifest.json").write_text("{}")
    calls = []
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/bin/{cmd}")
    monkeypatch.setattr(subprocess, "run", lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0))
    assert CliRunner().invoke(app, ["containerize", "q", "--tag", "me/q:1"]).exit_code == 0
    assert calls == [["/bin/docker", "build", "-f", str(MODEL_DOCKERFILE), "-t", "me/q:1", str(root / "export")]]
