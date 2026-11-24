from __future__ import annotations

import hashlib
import io
import shutil
import tarfile

import pytest
from typer.testing import CliRunner

from byosm.cli import app
from byosm.games.poker import sharingan
from byosm.games.poker.sharingan import weights

FOUND = sharingan.Blueprint("someone/some-blueprint", "abc123", "tiny.plcp", 2, "medium")
CHUNKS = [b"first half ", b"second half"]


def bucket_archive() -> bytes:
    held = io.BytesIO()
    with tarfile.open(fileobj=held, mode="w") as tar:
        table = b"card buckets"
        info = tarfile.TarInfo(f"{sharingan.BUCKETS}/flop.bin")
        info.size = len(table)
        tar.addfile(info, io.BytesIO(table))
    return held.getvalue()


@pytest.fixture
def hub(ws, tmp_path, monkeypatch):
    remote, asked = tmp_path / "hub", []
    remote.mkdir()
    files = dict(zip(FOUND.part_names(), CHUNKS, strict=True)) | {weights.ARCHIVE: bucket_archive()}
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    whole = hashlib.sha256(b"".join(CHUNKS)).hexdigest()
    files[weights.PART_SUMS] = "".join(f"{d}  {n}\n" for n, d in digests.items()).encode()
    files[weights.FILE_SUMS] = f"{whole}  {FOUND.file}\n".encode()
    for name, data in files.items():
        (remote / name).write_bytes(data)

    def download(found, folder, name):
        asked.append(name)
        (folder / weights.PARTS).mkdir(parents=True, exist_ok=True)
        return shutil.copy(remote / name, folder / weights.PARTS / name)

    monkeypatch.setitem(sharingan.BLUEPRINTS, "tiny", FOUND)
    monkeypatch.setattr(weights, "download", download)
    monkeypatch.setattr(weights, "sizes", lambda found: {n: len(d) for n, d in files.items()})
    return asked


def test_a_blueprint_is_fetched_joined_and_its_buckets_unpacked_once(ws, hub):
    made = weights.fetch(ws, "tiny", lambda _: None)
    root = sharingan.work_dir(ws)
    assert (root / sharingan.WEIGHTS / FOUND.file).read_bytes() == b"".join(CHUNKS)
    assert (root / sharingan.DATA / sharingan.BUCKETS / "flop.bin").read_bytes() == b"card buckets"
    assert made["blueprint"] == "tiny" and made["file"].endswith(FOUND.file)
    hub.clear()
    weights.fetch(ws, "tiny", lambda _: None)
    assert sorted(hub) == sorted([weights.PART_SUMS, weights.FILE_SUMS])


def test_a_fetch_refuses_a_disk_that_cannot_hold_the_blueprint(ws, hub, monkeypatch):
    usage = shutil.disk_usage(ws)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=int(5 * weights.GB)))
    with pytest.raises(RuntimeError, match="5 GB free"):
        weights.fetch(ws, "tiny", lambda _: None)
    assert not any(name in hub for name in FOUND.part_names())


def test_an_unknown_blueprint_is_refused_by_name(ws):
    result = CliRunner().invoke(app, ["poker", "blueprint-fetch", "nothing-like-it"])
    assert result.exit_code != 0 and "no published blueprint" in result.output
