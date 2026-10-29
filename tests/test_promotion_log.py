from byosm.events import JOB_ENV
from byosm.project import FROM_TERMINAL, Project
from byosm.spec import Label, TaskSpec


def test_every_promotion_is_logged_with_what_it_replaced_why_and_who(ws, monkeypatch):
    p = Project("t").ensure()
    p.save_spec(TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")]))
    p.promote("v1", "first model")
    monkeypatch.setenv(JOB_ENV, "job-7")
    p.promote("v2", "sure lead on calib")
    log = p.promotions()
    assert [(r["before"], r["after"], r["why"], r["by"]) for r in log] == [
        (None, "v1", "first model", FROM_TERMINAL),
        ("v1", "v2", "sure lead on calib", "job-7"),
    ]
    assert p.promoted == "v2" and log[0]["at"] <= log[1]["at"]
