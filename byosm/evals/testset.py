from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from ..config import TESTSET_PER_LABEL, TESTSET_SUITE, load_config
from ..data.judge import JUDGE_ROLE, judge
from ..data.store import STORED_SPLITS, Dataset, append, dedupe, label_counts, read
from ..data.synth_text import generate
from ..db.rows import splits
from ..events import stage
from ..llm import role_model
from ..project import Project
from ..spec import EvalSuite, TaskSpec
from .suites import CALIB_EVERY_NTH_ITEM, freeze, suite_dir

WRITER_ROLE = "eval"
WRITTEN_SPLIT = "test"
OVERSAMPLE = 2
MAX_ROUNDS = 4
WRITTEN_MARK = ".written"
WRITTEN_SUFFIX = f"{WRITTEN_MARK}.jsonl"
LAYOUT_EXAMPLES = 3
LAYOUT_CHARS = 500


def written_path(p: Project, suite: str) -> Path:
    return p.evals_dir / f"{suite}{WRITTEN_SUFFIX}"


def written_logs(p: Project) -> dict[str, list[dict]]:
    logs = [n for n in splits(p.evals_dir, newest_first=True) if n.endswith(WRITTEN_MARK)]
    return {n.removesuffix(WRITTEN_MARK): read(p.evals_dir / f"{n}.jsonl") for n in logs}


def written_logger(path: Path) -> Callable[[list[dict]], None]:
    lock = threading.Lock()

    def add(rows: list[dict]) -> None:
        with lock:
            append(path, rows)

    return add


def _resumed(spec: TaskSpec, path: Path, against: list[dict], cfg: dict, log: Callable[[str], None]) -> list[dict]:
    rows = dedupe(read(path), against=against)
    if rows:
        log(f"  resuming from {len(rows)} inputs already written to {path.name}")
    return judge(spec, rows, cfg, log=log) if rows and spec.data.judge else rows


def existing_rows(p: Project, spec: TaskSpec) -> list[dict]:
    ds = Dataset(p.data_dir_for(spec))
    return [r for split in STORED_SPLITS for r in ds.load(split)]


def with_layout(spec: TaskSpec, rows: list[dict]) -> TaskSpec:
    if spec.input_description or not rows:
        return spec
    shown = "\n---\n".join(str(r["input"])[:LAYOUT_CHARS] for r in rows[:LAYOUT_EXAMPLES])
    layout = f"Write every input in exactly the layout of these real ones, with the same separators and field order:\n{shown}"
    return spec.model_copy(update={"input_description": layout})


def _short(spec: TaskSpec, rows: list[dict], per_label: int) -> dict[str, int]:
    counts = label_counts(rows)
    return {lab: per_label - counts.get(lab, 0) for lab in spec.label_names if counts.get(lab, 0) < per_label}


def _collect(
    spec: TaskSpec, per_label: int, against: list[dict], cfg: dict, log: Callable[[str], None], path: Path
) -> list[dict]:
    kept = _resumed(spec, path, against, cfg, log)
    on_rows = written_logger(path)
    for _ in range(MAX_ROUNDS):
        short = _short(spec, kept, per_label)
        if not short:
            break
        n = sum(short.values()) * OVERSAMPLE
        kept += generate(spec, n, WRITTEN_SPLIT, cfg, [*against, *kept], log=log, labels=list(short), on_rows=on_rows)
    return kept


def _calib_test(spec: TaskSpec, rows: list[dict], per_label: int) -> tuple[list[dict], list[dict]]:
    calib, test = [], []
    for lab in spec.label_names:
        for i, r in enumerate([r for r in rows if r["label"] == lab][:per_label]):
            (calib if i % CALIB_EVERY_NTH_ITEM == 0 else test).append(r)
    return calib, test


def _check(spec: TaskSpec, suite: str) -> None:
    if spec.task_type != "classify":
        raise ValueError(f"{spec.name} is a {spec.task_type} task; test sets can be written for classification only")
    if any(e.name == suite for e in spec.evals):
        raise ValueError(f"suite {suite} exists; pass another name with --suite")


@stage("Write test inputs")
def write_testset(
    p: Project,
    spec: TaskSpec,
    per_label: int = TESTSET_PER_LABEL,
    suite: str = TESTSET_SUITE,
    cfg: dict | None = None,
    log: Callable[[str], None] = print,
) -> EvalSuite:
    _check(spec, suite)
    cfg = cfg or load_config()
    known = existing_rows(p, spec)
    rows = _collect(with_layout(spec, known), per_label, known, cfg, log, written_path(p, suite))
    calib, test = _calib_test(spec, rows, per_label)
    if not test:
        raise ValueError(f"the LLM wrote no usable examples for {spec.name}")
    if short := _short(spec, calib + test, per_label):
        log(f"  short of {per_label} per label after {MAX_ROUNDS} rounds: {short}")
    judge = role_model(JUDGE_ROLE, cfg) if spec.data.judge else None
    es = EvalSuite(name=suite, kind="generated", written_by=role_model(WRITER_ROLE, cfg), judged_by=judge)
    freeze(suite_dir(p, es), es, calib, test)
    fresh = p.spec()
    fresh.evals.append(es)
    p.save_spec(fresh)
    return es
