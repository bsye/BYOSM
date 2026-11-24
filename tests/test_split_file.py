from __future__ import annotations

import hashlib
import shutil
import threading

import pytest

from byosm import split_file
from byosm.split_file import joined, reporting, sums

NAME = "big.bin"
PARTS = [f"{NAME}.part-{i:02d}" for i in range(3)]


@pytest.fixture
def remote(tmp_path):
    folder = tmp_path / "remote"
    folder.mkdir()
    chunks = [bytes([i]) * (1000 + i) for i in range(3)]
    for name, chunk in zip(PARTS, chunks, strict=True):
        (folder / name).write_bytes(chunk)
    whole = b"".join(chunks)
    published = {name: hashlib.sha256(chunk).hexdigest() for name, chunk in zip(PARTS, chunks, strict=True)}
    return folder, whole, {**published, NAME: hashlib.sha256(whole).hexdigest()}


def downloader(remote_folder, folder, asked, fail_on=None):
    def download(part):
        asked.append(part)
        if part == fail_on:
            raise OSError("the connection dropped")
        (folder / "parts").mkdir(parents=True, exist_ok=True)
        return shutil.copy(remote_folder / part, folder / "parts" / part)

    return download


def test_published_sums_are_read_by_file_name():
    assert sums("ab12  one.bin\ncd34  two.bin\n\n") == {"one.bin": "ab12", "two.bin": "cd34"}


def test_parts_are_checked_joined_and_removed_and_a_second_call_downloads_nothing(tmp_path, remote):
    source, whole, published = remote
    folder, asked = tmp_path / "local", []
    final = joined(folder, NAME, PARTS, published, downloader(source, folder, asked), lambda _: None)
    assert final.read_bytes() == whole and asked == PARTS
    assert sorted(p.name for p in folder.iterdir()) == [NAME, "parts"] and not list((folder / "parts").iterdir())
    assert joined(folder, NAME, PARTS, published, downloader(source, folder, asked), lambda _: None) == final
    assert asked == PARTS


def test_an_interrupted_join_resumes_from_the_last_whole_part(tmp_path, remote):
    source, whole, published = remote
    folder, asked = tmp_path / "local", []
    with pytest.raises(OSError, match="dropped"):
        joined(folder, NAME, PARTS, published, downloader(source, folder, asked, PARTS[2]), lambda _: None)
    with (folder / f"{NAME}{split_file.PARTIAL}").open("ab") as torn:
        torn.write(b"half a part written before the crash")
    asked.clear()
    final = joined(folder, NAME, PARTS, published, downloader(source, folder, asked), lambda _: None)
    assert final.read_bytes() == whole and asked == [PARTS[2]]


def test_a_part_that_does_not_match_its_checksum_is_deleted_and_nothing_is_published(tmp_path, remote):
    source, _, published = remote
    folder = tmp_path / "local"
    wrong = {**published, PARTS[1]: "0" * 64}
    with pytest.raises(ValueError, match="not the published"):
        joined(folder, NAME, PARTS, wrong, downloader(source, folder, []), lambda _: None)
    assert not (folder / NAME).exists() and not (folder / "parts" / PARTS[1]).exists()


def test_a_whole_file_that_does_not_match_starts_again_from_nothing(tmp_path, remote):
    source, whole, published = remote
    folder, asked = tmp_path / "local", []
    with pytest.raises(ValueError, match="not the published"):
        joined(folder, NAME, PARTS, {**published, NAME: "0" * 64}, downloader(source, folder, asked), lambda _: None)
    assert not (folder / NAME).exists()
    asked.clear()
    assert (
        joined(folder, NAME, PARTS, published, downloader(source, folder, asked), lambda _: None).read_bytes() == whole
    )
    assert asked == PARTS


def test_progress_is_reported_while_the_work_runs(tmp_path):
    (tmp_path / "half").write_bytes(b"x" * 500)
    said, heard = [], threading.Event()

    def log(line):
        said.append(line)
        heard.set()

    with reporting(tmp_path, 1000, log, every=0.01):
        assert heard.wait(5)
    assert said[0] == "progress: 0.0 of 0.0 GB on disk"
