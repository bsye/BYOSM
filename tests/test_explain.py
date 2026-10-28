import pytest
from fastapi.testclient import TestClient

from byosm.app import bundles, create_app
from byosm.app.explain import MAX_STEPS, _cuts, explain

AUTH = {"Authorization": "Bearer t"}


class Model:
    labels = ["neg", "pos"]

    def __init__(self, modality="text"):
        self.m = {"positive_index": 1, "decision": {"threshold": 0.7}, "task_type": "classify", "modality": modality}

    def predict(self, text):
        p = 0.9 if "good" in text.split() else 0.2
        return {"label": "pos" if p >= 0.7 else "neg", "probs": {"neg": round(1 - p, 4), "pos": p}}


def test_the_reading_follows_the_answer_word_by_word_and_the_driver_is_the_word_that_moves_it():
    got = explain(Model(), "this is good stuff")
    assert got["label"] == got["follows"] == "pos"
    assert got["words"] == ["this", "is", "good", "stuff"]
    assert [r["p"] for r in got["reading"]] == [0.2, 0.2, 0.9, 0.9]
    assert max(got["drivers"], key=lambda d: d["drop"])["word"] == "good"
    assert got["threshold"] == 0.7


def test_following_the_first_answer_keeps_the_curve_comparable_with_the_threshold_of_the_positive_label():
    got = explain(Model(), "this is bad", follow="pos")
    assert got["label"] == "neg" and got["follows"] == "pos"
    assert got["reading"][-1]["p"] == 0.2 and got["threshold"] == 0.7
    assert explain(Model(), "this is good", follow="neg")["threshold"] is None
    with pytest.raises(ValueError):
        explain(Model(), "x", follow="maybe")


def test_long_inputs_are_read_in_at_most_max_steps_cuts_ending_on_the_last_word():
    cuts = _cuts(1000)
    assert len(cuts) <= MAX_STEPS + 1 and cuts[-1] == 1000
    assert _cuts(3) == [1, 2, 3]


def test_the_route_explains_text_classifiers_only(ws, monkeypatch):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        url = "/api/specialists/q/explain"
        assert c.post(url, json={"input": "good"}, headers=AUTH).status_code == 404
        monkeypatch.setattr(bundles, "bundle", lambda ws, name: Model())
        assert c.post(url, json={"input": "good"}, headers=AUTH).json()["label"] == "pos"
        assert c.post(url, json={"input": "good", "follow": "maybe"}, headers=AUTH).status_code == 422
        monkeypatch.setattr(bundles, "bundle", lambda ws, name: Model("audio"))
        assert c.post(url, json={"input": "good"}, headers=AUTH).status_code == 422
