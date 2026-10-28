import pytest

from byosm.lab.drafts import APPROVED, decide, drafts, save_draft
from byosm.project import Project
from byosm.scout.fulltext import method_text
from byosm.spec import Label, TaskSpec

CODE = "def rows(train, rng):\n    return train\n"
PAPER_HTML = """<html><h2>1 Introduction</h2><p>Why it matters.</p>
<h2>3 Method</h2><p>We train with a <b>margin</b> loss.</p><h3>3.1 Training recipe</h3><p>Warm up for 5% of steps.</p>
<h2>5 Results</h2><p>It wins.</p></html>"""


def test_only_the_method_and_training_sections_of_a_paper_are_kept():
    text = method_text(PAPER_HTML)
    assert "margin loss" in text and "Warm up" in text and "Why it matters" not in text and "It wins" not in text


def _project():
    p = Project("tc").ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name="tc", description="q", modality="text", labels=labels))
    return p


class Writer:
    def __init__(self):
        self.asked = []

    def json(self, system, prompt, schema):
        self.asked.append(prompt)
        return {"name": f"v{len(self.asked)}", "hypothesis": "h", "code": CODE + f"\nSTEP = {len(self.asked)}\n"}


def test_a_failed_run_writes_a_revision_that_waits_for_approval_until_two_revisions(ws, monkeypatch):
    from byosm import pipeline
    from byosm.lab import author
    from byosm.lab.run import run

    p = _project()
    writer = Writer()
    monkeypatch.setattr("byosm.llm.get_llm", lambda role: writer)
    monkeypatch.setattr(author, "_prompt", lambda p, instruction, paper, log: instruction)
    monkeypatch.setattr(pipeline, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("loss is nan")))
    current = save_draft(p, "first", "h", CODE, "test")
    for _ in range(3):
        decide(p, current["id"], APPROVED)
        with pytest.raises(RuntimeError):
            run("tc", current["id"], lambda _: None)
        current = drafts(p)[0]
    assert len(writer.asked) == 2 and all("loss is nan" in a for a in writer.asked)
    assert [d["status"] for d in drafts(p)] == ["approved", "approved", "approved"]
    assert drafts(p)[0]["parent"] == drafts(p)[1]["id"]


def test_the_author_reads_the_method_sections_of_the_paper_it_implements(ws, monkeypatch):
    from byosm.lab import author

    p = _project()
    monkeypatch.setattr("byosm.scout.fulltext.paper_method", lambda url, log: "## 3 Method\nmargin loss")
    assert author._paper("https://arxiv.org/abs/2610.00001", lambda _: None)[0].endswith("margin loss")
    assert author._paper("", lambda _: None) == []
    writer = Writer()
    monkeypatch.setattr("byosm.llm.get_llm", lambda role: writer)
    monkeypatch.setattr(author, "_prompt", lambda p, instruction, paper, log: paper)
    d = author.write("tc", "", lambda _: None, "https://arxiv.org/abs/2610.00001")
    assert d["paper"] == "https://arxiv.org/abs/2610.00001" and writer.asked == ["https://arxiv.org/abs/2610.00001"]
    assert p.report("lab:" + d["id"])["paper"] == d["paper"]
