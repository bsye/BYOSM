import numpy as np
import pytest
import yaml
from conftest import save_board
from typer.testing import CliRunner

from byosm.cli import app
from byosm.format import cpu, fmt, interval, params
from byosm.project import Project
from byosm.report import RESULTS_PAGE, blocks, fill, keys_in, replace_block, unexpected
from byosm.report.cells import num, score
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
    p.save_spec(
        TaskSpec(
            name="toy",
            description="toy task",
            summary="toy yes/no",
            modality="text",
            labels=[Label(name="no", definition="no"), Label(name="yes", definition="yes")],
            objective=Objective(metric="accuracy"),
            evals=[EvalSuite(name="bench", primary=True)],
            baselines=["rival"],
        )
    )
    for exp, parent in (("v1", None), ("v2", "v1")):
        applied = [{"id": "more-data", "why": "", "ops": []}]
        p.save_experiment(exp, meta={}, proposal={"parent": parent, "applied": applied} if parent else None)
    board = {
        "v1": {"bench": _result("experiment", 0.5, 0.5, BASE_UNITS, 2_000_000)},
        "v2": {"bench": _result("experiment", 5 / 6, 0.75, TEST_UNITS, 2_000_000)},
        "rival": {"bench": _result("baseline", 0.5, 0.5, BASE_UNITS, 9_000_000)},
    }
    save_board(p, board)
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


def _trained(p: Project, train: dict) -> TaskSpec:
    spec = p.spec()
    return spec.model_copy(update={"train": spec.train.model_copy(update=train)})


def test_built_from_names_only_lineage_and_folds(toy):
    p = Project("toy")
    p.save_experiment("v1", spec=_trained(p, {"epochs": 3}))
    p.save_experiment("v2", spec=_trained(p, {"epochs": 5, "ema": True}))
    p.save_experiment("v2", meta={"n_train": 120, "n_test_folded": 20})
    row = blocks()["toy"].split("\n")[2]
    assert "| v1 + more-data | 120 (+20 builtin test) |" in row and "differs" not in row
    assert "builtin suite's held-out split" in blocks()["toy"]


def test_summary_without_baseline(toy):
    p = Project("toy")
    p.drop_result("rival", "bench")
    assert "no outside baseline scored yet" in blocks()["summary"]


def test_replace_block_keeps_surroundings():
    text = "intro\n<!-- report:x -->\nold\n<!-- /report:x -->\noutro\n"
    assert replace_block(text, "x", "new") == "intro\n<!-- report:x -->\nnew\n<!-- /report:x -->\noutro\n"
    assert replace_block(text, "y", "new") is None
    new, missing = fill(text, {"x": "new", "y": "more"})
    assert missing == ["y"] and new.endswith("<!-- report:y -->\nmore\n<!-- /report:y -->\n")


def test_cli_writes_then_check_passes_and_fails_when_stale(toy):
    page = toy / RESULTS_PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "# toy\n\n<!-- report:summary -->\n<!-- /report:summary -->\n\n<!-- report:toy -->\n<!-- /report:toy -->\n"
    )
    runner = CliRunner()
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    assert runner.invoke(app, ["report"]).exit_code == 0
    assert "| **v2** (promoted)" in page.read_text() and page.read_text().startswith("# toy\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 0
    good = page.read_text()
    page.write_text(good.replace("0.833", "0.999"))
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    page.write_text(good + "\n<!-- report:gone -->\n| x | 0.999 |\n<!-- /report:gone -->\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1
    assert runner.invoke(app, ["report"]).exit_code == 1
    page.write_text(good + "\n<!-- report:toy -->\n| **v2** | 0.999 |\n<!-- /report:toy -->\n")
    assert runner.invoke(app, ["report", "--check"]).exit_code == 1


def test_unknown_and_repeated_blocks_are_flagged():
    text = "<!-- report:a -->\n<!-- /report:a -->\n<!-- report:b -->x<!-- /report:b --><!-- report:a -->"
    assert keys_in(text) == ["a", "b", "a"]
    assert unexpected(text, {"a": "", "b": ""}) == ["a"]
    assert unexpected(text, {"b": ""}) == ["a"]
    assert unexpected("<!-- report:b -->", {"a": "", "b": ""}) == []


def test_seed_groups_and_stale_rows(toy):
    p = Project("toy")
    board = p.board()
    for i, (acc, calib) in enumerate(((0.5, 0.6), (0.7, 0.8))):
        exp = f"rec-s{i}"
        p.exp_dir(exp).mkdir(parents=True, exist_ok=True)
        p.save_experiment(exp, spec=_trained(p, {"ema": True, "seed": i}), meta={})
        board[exp] = {"bench": _result("experiment", acc, calib, BASE_UNITS, 2_000_000) | {"fingerprint": "old"}}
    board["v2"]["bench"]["fingerprint"] = "new"
    save_board(p, board)
    seeds = blocks()["toy:seeds"]
    assert seeds.startswith(
        "| recipe | seeds | train rows | test accuracy % | calib accuracy % |\n|---|---|---|---|---|\n"
    )
    assert "| rec† | 2 | - | 60.00 ± 14.14 | 70.00 ± 14.14 |" in seeds
    assert "sample sd" in seeds and "no recipe has seeds on the promoted model's suite version" in seeds
    assert "| rec" not in blocks()["toy"] and "except 2 seed replicates" in blocks()["toy"]


def _replicates(p: Project, runs: tuple[tuple[float, float, str], ...]) -> None:
    board = p.board()
    for i, (acc, calib, fp) in enumerate(runs):
        exp = f"rec-s{i}"
        p.exp_dir(exp).mkdir(parents=True, exist_ok=True)
        p.save_experiment(exp, spec=_trained(p, {"seed": i}), meta={})
        board[exp] = {"bench": _result("experiment", acc, calib, BASE_UNITS, 2_000_000) | {"fingerprint": fp}}
    board["v2"]["bench"]["fingerprint"] = "new"
    save_board(p, board)


def test_a_lone_replicate_on_another_fingerprint_stays_in_the_main_table(toy):
    _replicates(Project("toy"), ((0.5, 0.6, "old"), (0.7, 0.8, "old"), (0.9, 0.9, "new")))
    seeds, main = blocks()["toy:seeds"], blocks()["toy"]
    assert "| rec† | 2 | - | 60.00 ± 14.14 | 70.00 ± 14.14 |" in seeds
    assert "rec-s2" not in seeds and "top on calib" not in seeds
    assert "| rec-s2 |" in main and "except 2 seed replicates" in main


def test_each_fingerprint_subgroup_gets_its_own_recipe_row(toy):
    _replicates(Project("toy"), ((0.5, 0.5, "old"), (0.6, 0.6, "old"), (0.8, 0.8, "new"), (0.9, 0.9, "new")))
    seeds = blocks()["toy:seeds"]
    assert "| rec @old† | 2 | - | 55.00 ± 7.07 | 55.00 ± 7.07 |" in seeds
    assert "| rec @new | 2 | - | 85.00 ± 7.07 | 85.00 ± 7.07 |" in seeds
    assert "top on calib: `rec @new`" in seeds and "except 4 seed replicates" in blocks()["toy"]


def test_format_cells():
    assert fmt(None) == "–" and fmt(None, missing="-") == "-" and fmt(0.12345) == "0.1235" and fmt(7) == "7.0000"
    assert params(2_345_678) == "2.35M" and params(None) == "-" and params(0, "–") == "–"
    assert cpu({"stream_ms": 0.1694, "chunk_ms": 30}) == "0.169/30ms"
    assert cpu({"latency_ms": 3.371}) == "3.37ms" and cpu({}) == "-"
    assert interval({"lo": 0.8631, "hi": 0.899}) == "0.863-0.899" and interval(None) == ""
    assert interval({"lo": 0.5, "hi": 0.75}, 4, ", ") == "0.5000, 0.7500"


def test_format_numbers_whatever_their_type():
    assert fmt(1, 3) == "1.000" and fmt(0) == "0.0000" and fmt(np.float32(0.5), 3) == "0.500"
    assert fmt(True) == "True" and fmt("n/a") == "n/a"
    assert num(1) == "1.000" and score(1, {"lo": 0.9, "hi": 1}) == "1.000 [0.900, 1.000]"


def test_a_stale_recipe_is_never_ranked_top(toy):
    _replicates(Project("toy"), ((0.9, 0.95, "old"), (0.9, 0.97, "old"), (0.8, 0.8, "new"), (0.81, 0.81, "new")))
    seeds = blocks()["toy:seeds"]
    assert seeds.index("| rec @new |") < seeds.index("| rec @old† |")
    assert "top on calib: `rec @new`" in seeds
