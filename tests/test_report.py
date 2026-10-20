import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from byosm.cli import app
from byosm.config import Project
from byosm.report import blocks, fill, keys_in, replace_block, sync, unexpected
from byosm.spec import TaskSpec
from byosm.spec.sections import EvalSuite, Label, Objective

TEST_UNITS = {"y": [0, 1, 0, 1, 1, 0], "pred": [0, 1, 0, 1, 0, 0]}
BASE_UNITS = {"y": [0, 1, 0, 1, 1, 0], "pred": [1, 1, 0, 0, 0, 0]}


def _result(kind: str, objective: float, calib: float, units: dict, params: int) -> dict:
    return {
        "kind": kind,
        "params": params,
        "objective": objective,
        "calib_objective": calib,
        "latency_ms": 1.5,
        "n_test": 6,
        "n_calib": 4,
        "units": {"test": units},
        "metrics": {"ci95": {"lo": objective - 0.1, "hi": min(1.0, objective + 0.1)}},
    }


@pytest.fixture
def toy(tmp_path, monkeypatch):
    (tmp_path / "byosm.yaml").write_text(yaml.safe_dump({"llm": {"default": {"provider": "fake"}}}))
    monkeypatch.chdir(tmp_path)
    p = Project("toy").ensure()
    TaskSpec(
        name="toy",
        description="toy task",
        summary="toy yes/no",
        modality="text",
        labels=[Label(name="no", definition="no"), Label(name="yes", definition="yes")],
        objective=Objective(metric="accuracy"),
        evals=[EvalSuite(name="bench", primary=True)],
        baselines=["rival"],
    ).save(p.spec_path)
    for exp, parent in (("v1", None), ("v2", "v1")):
        p.exp_dir(exp).mkdir(parents=True, exist_ok=True)
        if parent:
            proposal = {"parent": parent, "applied": [{"id": "more-data", "why": "", "ops": []}]}
            (p.exp_dir(exp) / "proposal.json").write_text(json.dumps(proposal))
    board = {
        "v1": {"bench": _result("experiment", 0.5, 0.5, BASE_UNITS, 2_000_000)},
        "v2": {"bench": _result("experiment", 5 / 6, 0.75, TEST_UNITS, 2_000_000)},
        "rival": {"bench": _result("baseline", 0.5, 0.5, BASE_UNITS, 9_000_000)},
    }
    p.board_path.write_text(json.dumps(board))
    p.promote("v2")
    return tmp_path


def test_blocks_render_board(toy):
    out = blocks()
    assert set(out) == {"summary", "toy"}
    assert "| toy yes/no | toy-bench (accuracy) | v2 (2.00M) | 0.833 [0.733, 0.933] | rival (9.00M) |" in out["summary"]
    assert "+0.333, P(better)" in out["summary"] and "| 0.750 vs 0.500 (ahead) |" in out["summary"]
    assert "| **v2** (promoted) | v1 + more-data | - | 2.00M | 1.50ms | 0.833 [0.733, 0.933] | 0.750 |" in out["toy"]
    assert "| rival | outside baseline | - |" in out["toy"] and "| v1 | no recorded parent |" in out["toy"]
    assert "6 test and 4 calib items" in out["toy"] and "Every scored experiment is listed." in out["toy"]


def test_built_from_names_unexplained_fields_and_folds(toy):
    p = Project("toy")
    (p.exp_dir("v1") / "spec.yaml").write_text(yaml.safe_dump({"train": {"epochs": 3}}))
    (p.exp_dir("v2") / "spec.yaml").write_text(yaml.safe_dump({"train": {"epochs": 5, "ema": True}}))
    (p.exp_dir("v2") / "meta.json").write_text(json.dumps({"n_train": 120, "n_test_folded": 20}))
    row = blocks()["toy"].split("\n")[2]
    assert "v1 + more-data; spec also differs in 2 other fields (train) | 120 (+20 builtin test) |" in row
    assert "builtin suite's held-out split" in blocks()["toy"]


def test_summary_without_baseline(toy):
    p = Project("toy")
    board = json.loads(p.board_path.read_text())
    del board["rival"]
    p.board_path.write_text(json.dumps(board))
    assert "no outside baseline scored yet" in blocks()["summary"]


def test_replace_block_keeps_surroundings():
    text = "intro\n<!-- report:x -->\nold\n<!-- /report:x -->\noutro\n"
    assert replace_block(text, "x", "new") == "intro\n<!-- report:x -->\nnew\n<!-- /report:x -->\noutro\n"
    assert replace_block(text, "y", "new") is None
    new, missing = fill(text, {"x": "new", "y": "more"})
    assert missing == ["y"] and new.endswith("<!-- report:y -->\nmore\n<!-- /report:y -->\n")


def test_cli_writes_then_check_passes_and_fails_when_stale(toy):
    readme = toy / "README.md"
    readme.write_text(
        "# toy\n\n<!-- report:summary -->\n<!-- /report:summary -->\n\n<!-- report:toy -->\n<!-- /report:toy -->\n"
    )
    runner = CliRunner()
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    assert runner.invoke(app, ["report"]).exit_code == 0
    assert "| **v2** (promoted)" in readme.read_text() and readme.read_text().startswith("# toy\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 0
    good = readme.read_text()
    readme.write_text(good.replace("0.833", "0.999"))
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    readme.write_text(good + "\n<!-- report:gone -->\n| x | 0.999 |\n<!-- /report:gone -->\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    assert runner.invoke(app, ["report"]).exit_code == 1
    readme.write_text(good + "\n<!-- report:toy -->\n| **v2** | 0.999 |\n<!-- /report:toy -->\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1


def test_unknown_and_repeated_blocks_are_flagged():
    text = "<!-- report:a -->\n<!-- /report:a -->\n<!-- report:b -->x<!-- /report:b --><!-- report:a -->"
    assert keys_in(text) == ["a", "b", "a"]
    assert unexpected(text, {"a": "", "b": ""}) == ["a"]
    assert unexpected(text, {"b": ""}) == ["a"]
    assert unexpected("<!-- report:b -->", {"a": "", "b": ""}) == []


def test_seed_groups_and_stale_rows(toy):
    p = Project("toy")
    board = json.loads(p.board_path.read_text())
    for i, (acc, calib) in enumerate(((0.5, 0.6), (0.7, 0.8))):
        exp = f"rec-s{i}"
        p.exp_dir(exp).mkdir(parents=True, exist_ok=True)
        (p.exp_dir(exp) / "spec.yaml").write_text(yaml.safe_dump({"train": {"ema": True, "seed": i}}))
        board[exp] = {"bench": _result("experiment", acc, calib, BASE_UNITS, 2_000_000) | {"fingerprint": "old"}}
    board["v2"]["bench"]["fingerprint"] = "new"
    p.board_path.write_text(json.dumps(board))
    seeds = blocks()["toy:seeds"]
    assert seeds.startswith(
        "| recipe | seeds | train rows | test accuracy % | calib accuracy % |\n|---|---|---|---|---|\n"
    )
    assert "| rec† | 2 | - | 60.00 ± 14.14 | 70.00 ± 14.14 |" in seeds
    assert "sample sd" in seeds and "top on calib: `rec`" in seeds
    assert "| rec" not in blocks()["toy"] and "except 2 seed replicates" in blocks()["toy"]


def test_repository_readme_is_current(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    current, missing, extra = sync(root / "README.md", check=True)
    assert missing == [] and extra == [] and current, "README result tables are stale: run byosm report"
