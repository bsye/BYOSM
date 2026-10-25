import pytest
import yaml

from byosm.evals import board, run
from byosm.evals.identity import baseline_identity
from byosm.evals.suites import fingerprint
from byosm.llm import PROVIDERS, clear_cache
from byosm.project import Project
from byosm.spec import TaskSpec
from byosm.spec.sections import EvalSuite, Label, Objective

HF_REPO = "org/model"
LLM_CONF = {"provider": "fake", "model": "qwen3.8:27b", "reasoning_effort": "none"}


def _write_llm(ws, baseline=None, **conf):
    roles = {"default": {**LLM_CONF, **conf}, **({"baseline": baseline} if baseline else {})}
    (ws / "byosm.yaml").write_text(yaml.safe_dump({"llm": roles}))


class FakeLLM:
    def __init__(self, model, **_):
        self.name = f"fake:{model}"


class Scorer:
    def __init__(self, ws):
        self.ws = ws
        self.calls = []
        self.failing = set()
        self.downloads = False

    def __call__(self, pred, spec, calib, test, root, space=None):
        self.calls.append(pred)
        if self.downloads and pred.startswith("hf:"):
            _set_hf_revision(self.ws, "fresh")
        if pred in self.failing:
            raise RuntimeError("model gone")
        return {"kind": "baseline", "objective": 0.5, "calib_objective": 0.5, "metrics": {}}


@pytest.fixture
def toy(tmp_path, monkeypatch):
    _write_llm(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(PROVIDERS, "fake", FakeLLM)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
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
            baselines=["llm", f"hf:{HF_REPO}"],
        )
    )
    p.exp_dir("v1").mkdir(parents=True)
    p.save_experiment("v1", meta={})
    scorer = Scorer(tmp_path)
    monkeypatch.setattr(board, "materialize", lambda *a: ([{}], [{}], tmp_path))
    monkeypatch.setattr(board, "score", scorer)
    monkeypatch.setattr(board, "get_baseline", lambda b, spec: b)
    monkeypatch.setattr(board, "experiment_predictor", lambda p, base, e: e)
    return scorer


def _set_hf_revision(ws, sha):
    refs = ws / "hub" / f"models--{HF_REPO.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha)


def test_changing_the_llm_rescores_only_the_llm_baseline(toy, tmp_path):
    run("toy", log=lambda m: None)
    toy.calls.clear()
    _write_llm(tmp_path, model="qwen3.8:32b")
    run("toy", log=lambda m: None)
    assert toy.calls == ["llm"]
    _write_llm(tmp_path, model="qwen3.8:32b", reasoning_effort="high")
    run("toy", log=lambda m: None)
    assert toy.calls == ["llm", "llm"]


def test_hf_baselines_are_keyed_by_repo_not_by_the_local_hub_cache(toy, tmp_path):
    _set_hf_revision(tmp_path, "aaa")
    run("toy", log=lambda m: None)
    toy.calls.clear()
    _set_hf_revision(tmp_path, "bbb")
    run("toy", log=lambda m: None)
    assert toy.calls == []
    assert baseline_identity(f"hf:{HF_REPO}") == HF_REPO


def test_changing_the_default_under_a_partial_baseline_role_rescores_the_llm(toy, tmp_path):
    _write_llm(tmp_path, baseline={"model": "m1"})
    run("toy", log=lambda m: None)
    toy.calls.clear()
    _write_llm(tmp_path, baseline={"model": "m1"}, reasoning_effort="high", base_url="http://other")
    run("toy", log=lambda m: None)
    assert toy.calls == ["llm"]


def test_a_first_download_of_hf_weights_is_not_rescored_next_run(toy):
    toy.downloads = True
    run("toy", log=lambda m: None)
    assert f"hf:{HF_REPO}" in toy.calls
    toy.calls.clear()
    run("toy", log=lambda m: None)
    assert toy.calls == []


def test_existing_experiment_caches_stay_valid(toy):
    p = Project("toy")
    base = p.spec()
    fp = fingerprint(p, base, base.evals[0])
    p.save_result("v1", "bench", {"kind": "experiment", "objective": 0.9, "metrics": {}, "fingerprint": fp})
    results = run("toy", baselines=[], log=lambda m: None)
    assert toy.calls == []
    assert results["v1"]["bench"]["objective"] == 0.9


def test_a_failing_predictor_loses_its_stale_board_entry(toy):
    run("toy", log=lambda m: None)
    toy.failing.add("llm")
    logs = []
    results = run("toy", force=True, log=logs.append)
    assert "llm" not in results
    assert "llm" not in Project("toy").board()
    assert set(results) == {"v1", f"hf:{HF_REPO}"}
    assert any("llm" in m and "dropped" in m for m in logs)


class KeyedLLM(FakeLLM):
    def __init__(self, model, api_key_env, **_):
        import os

        os.environ[api_key_env]
        super().__init__(model)


def test_a_cached_llm_baseline_survives_a_machine_without_its_key(toy, tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDERS, "keyed", KeyedLLM)
    monkeypatch.setenv("BYOSM_TEST_KEY", "x")
    _write_llm(tmp_path, provider="keyed", api_key_env="BYOSM_TEST_KEY")
    run("toy", log=lambda m: None)
    clear_cache()
    monkeypatch.delenv("BYOSM_TEST_KEY")
    toy.calls.clear()
    results = run("toy", log=lambda m: None)
    assert toy.calls == []
    assert "llm" in results
    assert "llm" in Project("toy").board()


def _strip_identities(p):
    for pid, by_suite in p.board().items():
        for suite, r in by_suite.items():
            if r["kind"] == "baseline":
                p.save_result(pid, suite, {k: v for k, v in r.items() if k != "identity"})


def test_baseline_caches_from_before_identity_tracking_are_adopted_not_rescored(toy, tmp_path):
    _set_hf_revision(tmp_path, "aaa")
    run("toy", log=lambda m: None)
    _strip_identities(Project("toy"))
    toy.calls.clear()
    run("toy", log=lambda m: None)
    assert toy.calls == []
    _write_llm(tmp_path, model="qwen3.8:32b")
    run("toy", log=lambda m: None)
    assert toy.calls == ["llm"]


def test_plain_baselines_store_no_identity(toy):
    p = Project("toy")
    base = p.spec()
    p.save_spec(base.model_copy(update={"baselines": ["energy"]}))
    run("toy", log=lambda m: None)
    assert "identity" not in p.result("energy", "bench")


def test_retraining_an_experiment_rescores_it(toy):
    run("toy", baselines=[], log=lambda m: None)
    toy.calls.clear()
    run("toy", baselines=[], log=lambda m: None)
    assert toy.calls == []
    Project("toy").save_experiment("v1", meta={"best_epoch": 2})
    run("toy", baselines=[], log=lambda m: None)
    assert toy.calls == ["v1"]
