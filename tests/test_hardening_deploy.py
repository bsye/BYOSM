import contextlib
from types import SimpleNamespace

import byosm.deploy as deploy


def test_a_read_only_container_still_gets_a_writable_capped_tmp(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "_run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(deploy.urllib.request, "urlopen", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(deploy, "_free_port", lambda: 5000)
    deploy.deploy("Doc", log=lambda _: None)
    run = calls[-1]
    assert "--read-only" in run
    tmpfs = run[run.index("--tmpfs") + 1]
    assert tmpfs == deploy.TMPFS and tmpfs.startswith("/tmp:size=")
