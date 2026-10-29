import json

from binary_specs import Scripted, binary

from byosm.evals.scoring import SCORE_CHUNK, score
from byosm.events import EVENT_PREFIX, EVENTS_ENV


def _rows(n, split):
    return [
        {"id": f"{split}{i}", "input": str(i % 2), "label": "yes" if i % 2 else "no", "split": split} for i in range(n)
    ]


def test_scoring_reports_its_progress_per_chunk_of_rows_without_changing_the_score(tmp_path, monkeypatch, capsys):
    spec = binary("b", "no", "yes", "f1")
    calib, test = _rows(10, "calib"), _rows(SCORE_CHUNK + 44, "test")
    quiet = score(Scripted(spec, float), spec, calib, test, tmp_path, None, "builtin")
    monkeypatch.setenv(EVENTS_ENV, "1")
    loud = score(Scripted(spec, float), spec, calib, test, tmp_path, None, "builtin")
    events = [json.loads(line[len(EVENT_PREFIX) :]) for line in capsys.readouterr().out.splitlines()]
    seen = [(e["suite"], e["split"], e["done"], e["total"]) for e in events if e["kind"] == "scoring"]
    assert seen == [("builtin", "calib", 10, 10), ("builtin", "test", SCORE_CHUNK, 300), ("builtin", "test", 300, 300)]
    assert loud["objective"] == quiet["objective"] and loud["calib_objective"] == quiet["calib_objective"]
