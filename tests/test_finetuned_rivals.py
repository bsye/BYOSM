from byosm import finetuned_rivals as fr
from byosm.project import Project
from byosm.spec import EvalSuite, HFSource, Label, TaskSpec


def _spec(ws):
    p = Project("bank", ws)
    labels = [Label(name="card_lost", definition=""), Label(name="refund", definition="")]
    suite = EvalSuite(name="b77", kind="hf", primary=True, source=HFSource(dataset="mteb/banking77"))
    p.save_spec(TaskSpec(name="bank", description="d", modality="text", labels=labels, evals=[suite]))
    return p


def test_the_benchmark_is_searched_by_full_id_and_bare_name(ws):
    assert fr.benchmarks(_spec(ws).spec()) == ["mteb/banking77", "banking77"]


def test_llm_picks_rank_first_and_rivals_with_our_labels_are_added(ws, monkeypatch):
    import byosm.discover as discover

    p = _spec(ws)
    monkeypatch.setattr(fr, "candidates", lambda spec, log: ["org/named", "org/other"])
    labels = {"org/named": ["card_lost", "refund"], "org/other": ["sports", "news"]}
    monkeypatch.setattr(discover, "_model_labels", lambda repo: labels[repo])
    added = fr.add_finetuned("bank", log=lambda m: None)
    assert [a["rival"] for a in added] == ["hf:org/named"]
    assert "hf:org/named" in p.spec().baselines and p.report(fr.REPORT)["added"] == added
