from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from pathlib import Path

from ..config import Project
from ..data.store import Dataset, read, record, write
from ..spec import EvalSuite, TaskSpec

SCORE_VERSION = "3"
FINGERPRINT_CHARS_PER_FILE = 100000
MIN_CALIB_FILES = 8
CALIB_EVERY_NTH_ITEM = 5


def suite_dir(p: Project, suite: EvalSuite) -> Path:
    return p.evals_dir / suite.name


def fingerprint(p: Project, spec: TaskSpec, suite: EvalSuite) -> str:
    d = suite_dir(p, suite)
    blob = suite.model_dump_json()
    if d.exists():
        blob += "".join(f.read_text()[:FINGERPRINT_CHARS_PER_FILE] for f in sorted(d.glob("*.jsonl")))
    if suite.kind == "builtin":
        blob += spec.data_key()
    blob += spec.decision.model_dump_json() + SCORE_VERSION
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def materialize(
    p: Project, base: TaskSpec, suite: EvalSuite, log: Callable[[str], None] = print
) -> tuple[list[dict], list[dict], Path]:
    if suite.kind == "builtin":
        dd = p.data_dir_for(base)
        ds = Dataset(dd)
        return ds.load("dev"), ds.load("test"), dd
    d = suite_dir(p, suite)
    if (d / "test.jsonl").exists():
        return read(d / "calib.jsonl"), read(d / "test.jsonl"), d
    d.mkdir(parents=True, exist_ok=True)
    calib, test, suite = _fetch(p, base, suite, d, log)
    write(d / "calib.jsonl", calib)
    write(d / "test.jsonl", test)
    (d / "suite.json").write_text(suite.model_dump_json(indent=2))
    return calib, test, d


def _fetch(
    p: Project, base: TaskSpec, suite: EvalSuite, d: Path, log: Callable[[str], None]
) -> tuple[list[dict], list[dict], EvalSuite]:
    if suite.kind == "local":
        return *_split_local(suite), suite
    if suite.kind == "hf" and suite.source and suite.source.format == "segment_files":
        return *_fetch_segments(p, base, suite, log), suite
    if suite.kind == "hf" and suite.source:
        return _fetch_hf(base, suite, d, log)
    raise ValueError(f"suite {suite.name}: unsupported kind/source")


def _split_local(suite: EvalSuite) -> tuple[list[dict], list[dict]]:
    calib, test = [], []
    for r in read(Path(suite.path)):
        (calib if r.get("split") == "calib" else test).append(r)
    return calib, test


def _fetch_segments(
    p: Project, base: TaskSpec, suite: EvalSuite, log: Callable[[str], None]
) -> tuple[list[dict], list[dict]]:
    from ..data import segments
    from ..modalities.audio import load_wav

    src = suite.source
    rate = base.data.audio.sample_rate
    hop = int(rate * base.frame_hop_ms / 1000)
    n_cal = max(MIN_CALIB_FILES, src.max_files // 4)
    calib, test = [], []
    for part, n, rows in (("calib", n_cal, calib), ("test", src.max_files, test)):
        for f in segments.fetch(src, (part,), n, p.root / "cache"):
            n_frames = len(load_wav(f, rate)) // hop
            labels = segments.frame_labels(f, n_frames, base).tolist()
            rows.append(record(str(f), labels, f"hf:{src.dataset}", file=f.name))
    log(f"  suite {suite.name}: {len(calib)} calib + {len(test)} test files from {src.dataset}")
    return calib, test


def _fetch_hf(
    base: TaskSpec, suite: EvalSuite, d: Path, log: Callable[[str], None]
) -> tuple[list[dict], list[dict], EvalSuite]:
    from ..data import _save_media, hf_source

    tmp = base.model_copy(deep=True)
    tmp.data.hf_sources = [suite.source]
    src = (hf_source.resolve(tmp, None, log) or [suite.source])[0]
    rows = [(x, y) for x, y in hf_source.take(src, base, suite.max_items * 2) if y]
    random.Random(0).shuffle(rows)
    calib, test = [], []
    for i, (x, y) in enumerate(rows[: suite.max_items]):
        inp = x if base.modality == "text" else _save_media(base, d, "media", x)
        (calib if i % CALIB_EVERY_NTH_ITEM == 0 else test).append(record(inp, y, f"hf:{src.dataset}"))
    return calib, test, suite.model_copy(update={"source": src})
