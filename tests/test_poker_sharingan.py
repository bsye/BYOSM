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


SAID = [
    "500 deals, 3000 hands",
    "  byosm                  raw +90.0 ± 400.0 mbb/g  aivat +120.5 ± 60.0  var-ratio hand 2.6x deal 1.9x",
    "    by seat (SB,BB,..): -6948 +15270 +14571 +9189 +12878 +9108",
    "  compact                raw -18.0 ± 80.0 mbb/g  aivat -24.1 ± 12.0  var-ratio hand 2.6x deal 1.9x",
    'report byosm: {"bot":"byosm","fallbacks":2,"requests":4000,"restarts":0}',
]


@pytest.fixture
def table(ws, monkeypatch):
    from byosm.games.poker.sharingan import arena

    asked = []
    monkeypatch.setattr(arena, "limits", lambda: ("4", "8g"))
    monkeypatch.setattr(arena, "ensure_image", lambda log: None)
    monkeypatch.setattr(arena, "seated", lambda ws, student: f"ext:byosm=[{student}]")
    monkeypatch.setattr(arena, "streamed", lambda command, turn, who, args, log: asked.append(command) or SAID)
    return arena, asked


def test_a_student_is_seated_against_five_copies_of_a_blueprint_in_a_sealed_container(ws, table):
    from byosm.db.workspace import workspace_reports

    arena, asked = table
    with pytest.raises(ValueError, match="blueprint-fetch"):
        arena.versus(ws, "my-student", "sharingan-2", 500, lambda _: None)
    root = sharingan.work_dir(ws)
    (root / sharingan.WEIGHTS).mkdir(exist_ok=True)
    (root / sharingan.WEIGHTS / "sharingan-2.plcp").write_bytes(b"")
    (root / sharingan.DATA / sharingan.BUCKETS).mkdir(parents=True)
    made = arena.versus(ws, "my-student", "sharingan-2", 500, lambda _: None)
    sent = asked[0]
    played = sent[sent.index("bench") :]
    assert played[:6] == ["bench", "match", "--a", "ext:byosm=[my-student]", "--b", "compact:/weights/sharingan-2.plcp"]
    assert played[6:14] == [
        "--players",
        "6",
        "--deals",
        "500",
        "--bucketer",
        "tables:/data/buckets-200",
        "--sizing",
        "medium",
    ]
    assert "none" in sent and "--read-only" in sent and f"{root / sharingan.WEIGHTS}:/weights:ro" in sent
    assert made["mbb_per_hand"] == 120.5 and made["ci95"] == 60.0 and made["corrected"] and made["hands"] == 3000
    assert made["raw"] == {"mbb_per_hand": 90.0, "ci95": 400.0} and made["fallbacks"] == 2
    assert workspace_reports(ws, "poker-table:")["poker-table:my-student:vs:sharingan-2"]["field"] == "sharingan-2"


def test_a_table_where_our_seat_often_failed_to_answer_is_not_scored(ws, table, monkeypatch):
    arena, _ = table
    monkeypatch.setattr(arena, "MOST_FALLBACKS", 0.0001)
    with pytest.raises(RuntimeError, match="failed to answer 2 of 4000"):
        arena.played(ws, "my-student", "station", 500, lambda _: None)


def test_the_gauntlet_says_which_published_scores_it_reproduced(ws, monkeypatch):
    from byosm.db.workspace import workspace_reports
    from byosm.games.poker.sharingan import gauntlet

    scores = {"random": 3950.0, "nit": 420.0}
    monkeypatch.setattr(
        gauntlet, "played", lambda ws, hero, field, deals, log: {"mbb_per_hand": scores.get(field, 0.0), "ci95": 30.0}
    )
    said = []
    made = gauntlet.gauntlet(ws, "sharingan-2", 100, said.append)
    assert list(made["fields"]) == list(gauntlet.HOUSE) and made["hands"] == 600
    assert made["checks"]["random"]["reproduced"] and not made["checks"]["nit"]["reproduced"]
    assert any("five nit" in line and "NOT reproduced" in line for line in said)
    assert workspace_reports(ws, "poker-gauntlet:")["poker-gauntlet:sharingan-2"]["deals"] == 100
    assert gauntlet.gauntlet(ws, "my-student", 100, said.append)["checks"] == {}
