import json
from types import SimpleNamespace

from byosm.scout.papers import arxiv, daily
from byosm.scout.proposals import ACCEPTED, DAY_S, decide, due, scan, state
from byosm.scout.schedule import tick

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2610.00001v1</id><title>Contrastive  heads for tiny encoders</title>
<summary>We show a 4 point gain on Banking77 at 8 rows per label.</summary></entry>
<entry><id>http://arxiv.org/abs/2610.00002v1</id><title>Another paper</title><summary>Text.</summary></entry>
</feed>"""
DAILY = json.dumps([{"paper": {"id": "2610.00003", "title": "Patch masking for forecasters", "summary": "CRPS -5%"}}])


def fake_get(url):
    return DAILY.encode() if "huggingface" in url else ATOM.encode()


def test_papers_are_read_from_arxiv_and_the_hugging_face_daily_list():
    found = arxiv("text", "q", fake_get)
    assert [p.url for p in found] == ["https://arxiv.org/abs/2610.00001v1", "https://arxiv.org/abs/2610.00002v1"]
    assert found[0].title == "Contrastive heads for tiny encoders"
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
    assert state(ws)["last_scan"] == {"read": 3, "proposed": 4, "kept": 2}
    assert '"id": "setfit"' in llm.prompts[0]
    said = []
    assert scan(ws, said.append, llm, fake_get) == [] and len(llm.prompts) == 1
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
