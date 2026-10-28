import json
from types import SimpleNamespace

from byosm.scout.papers import arxiv, daily
from byosm.scout.proposals import ACCEPTED, DAY_S, decide, due, scan, state
from byosm.scout.schedule import tick

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2610.00001v1</id><published>2026-10-21T17:00:00Z</published><title>Contrastive  heads for tiny encoders</title>
<summary>We show a 4 point gain on Banking77 at 8 rows per label.</summary></entry>
<entry><id>http://arxiv.org/abs/2610.00002v1</id><title>Another paper</title><summary>Text.</summary></entry>
</feed>"""
DAILY = json.dumps([{"paper": {"id": "2610.00003", "title": "Patch masking for forecasters", "summary": "CRPS -5%"}}])


def fake_get(url):
    return DAILY.encode() if "huggingface" in url else ATOM.encode()


def test_papers_are_read_from_arxiv_and_the_hugging_face_daily_list():
    found = arxiv("text", "q", fake_get)
    assert [p.url for p in found] == ["https://arxiv.org/abs/2610.00001v1", "https://arxiv.org/abs/2610.00002v1"]
    assert found[0].title == "Contrastive heads for tiny encoders" and found[0].published == "2026-10-21"
    assert daily(fake_get)[0].url == "https://arxiv.org/abs/2610.00003"


class Reader:
    def __init__(self):
        self.prompts = []

    def json(self, system, prompt, schema):
        self.prompts.append(prompt)
        good = {
            "paper": "http://arxiv.org/abs/2610.00001",
            "title": "Contrastive head",
            "effect": "e",
            "kinds": ["text"],
        }
        made_up = {"paper": "https://arxiv.org/abs/9999.99999", "title": "Invented", "effect": "e"}
        known = {"paper": "https://arxiv.org/abs/2610.00003", "title": "Masking", "overlaps": "forecaster-pretraining"}
        fresh = {
            "paper": "https://arxiv.org/abs/2610.00002v1",
            "title": "Another idea",
            "effect": "e",
            "overlaps": "none",
        }
        return {"proposals": [good, made_up, known, fresh]}


def test_a_scan_keeps_only_new_techniques_that_cite_a_paper_it_read_and_reads_each_paper_once(ws):
    llm = Reader()
    new = scan(ws, lambda _: None, llm, fake_get)
    assert [p["title"] for p in new] == ["Contrastive head", "Another idea"] and new[0]["status"] == "proposed"
    assert new[0]["published"] == "2026-10-21"
    assert state(ws)["last_scan"] == {"read": 3, "proposed": 8, "kept": 2}
    assert '"id": "setfit"' in llm.prompts[0]
    said = []
    asked = len(llm.prompts)
    assert scan(ws, said.append, llm, fake_get) == [] and len(llm.prompts) == asked
    assert state(ws)["last_scan"]["read"] == 0
    assert decide(ws, new[0]["id"], ACCEPTED)["status"] == ACCEPTED
    assert state(ws)["proposals"][0]["status"] == ACCEPTED


def test_the_schedule_queues_one_scan_when_the_last_is_older_than_the_setting(ws):
    submitted = []
    runner = SimpleNamespace(snapshot=lambda: [], submit=submitted.append)
    cfg = {"scout": {"every_days": 7}}
    assert tick(ws, runner, cfg, 0.0) and submitted == [["scout"]]
    scan(ws, lambda _: None, Reader(), fake_get)
    last = state(ws)["last_run"]
    assert not due(ws, cfg, last + 6 * DAY_S) and due(ws, cfg, last + 7 * DAY_S)
    busy = SimpleNamespace(
        snapshot=lambda: [SimpleNamespace(args=["scout"], status="running")], submit=submitted.append
    )
    assert not tick(ws, busy, cfg, last + 8 * DAY_S)


def test_a_citation_matches_its_paper_without_scheme_or_version():
    from byosm.scout.proposals import paper_key

    assert paper_key("https://arxiv.org/abs/2610.00001v2") == paper_key("http://www.arxiv.org/abs/2610.00001/")
    assert paper_key("https://huggingface.co/papers/2610.00001") != paper_key("https://arxiv.org/abs/2610.00001")


def test_trying_a_technique_sends_it_to_the_labs_of_the_specialists_that_need_it_most(ws):
    from conftest import promoted_specialist, save_board
    from fastapi.testclient import TestClient

    from byosm.app import create_app
    from byosm.db.workspace import save_workspace_report

    for name, score in (("a", 0.95), ("b", 0.5), ("c", 0.9)):
        p = promoted_specialist(name=name)
        save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": score, "n_test": 500, "n_calib": 200}}})
    text = {"id": "t1", "title": "Head", "paper": "https://arxiv.org/abs/1", "effect": "e", "kinds": ["text"]}
    series = {"id": "s1", "title": "DHF", "paper": "https://arxiv.org/abs/2", "effect": "e", "kinds": ["series"]}
    save_workspace_report(ws, "scout", {"last_run": 1, "seen": [], "proposals": [text, series]})
    submitted = []
    app = create_app(ws, "")
    app.state.runner.submit = lambda args: submitted.append(args)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        sent = c.post("/api/scout/t1/try").json()["sent"]
        assert c.post("/api/scout/s1/try").status_code == 422
    assert len(sent) == 2 and "b" in sent and [a[:3] for a in submitted] == [["lab", "write", n] for n in sorted(sent)]
    assert all('Implement "Head"' in a[4] and a[5:] == ["--paper", "https://arxiv.org/abs/1"] for a in submitted)


def test_a_forecasting_technique_goes_to_the_labs_of_the_forecasters_with_the_lowest_health(ws, monkeypatch):
    from conftest import save_forecaster
    from fastapi.testclient import TestClient

    from byosm.app import create_app
    from byosm.db.workspace import save_workspace_report

    monkeypatch.setattr("byosm.series.published.scores", lambda config, naive_mase: {})
    series = {"id": "s1", "title": "DHF", "paper": "https://arxiv.org/abs/2", "effect": "e", "kinds": ["series"]}
    audio = {"id": "a1", "title": "VAD", "paper": "https://arxiv.org/abs/3", "effect": "e", "kinds": ["audio"]}
    save_workspace_report(ws, "scout", {"last_run": 1, "seen": [], "proposals": [series, audio]})
    submitted = []
    app = create_app(ws, "")
    app.state.runner.submit = lambda args: submitted.append(args)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.post("/api/scout/s1/try").status_code == 422
        for domain, ours in (("ahead", 0.7), ("behind", 0.95), ("near", 0.82)):
            save_forecaster(ws, domain, ours=ours, rival=0.8)
        assert c.post("/api/scout/s1/try").json()["sent"] == ["behind", "near"]
        assert c.post("/api/scout/a1/try").status_code == 422
    assert [a[:3] for a in submitted] == [["lab", "write", "behind"], ["lab", "write", "near"]]
    assert 'Implement "DHF"' in submitted[0][4]
