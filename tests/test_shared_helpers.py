import json

from byosm.config import Project
from byosm.spec import Label, TaskSpec
from byosm.text import normalize


def test_normalize_folds_case_punctuation_underscores_and_spaces():
    assert normalize("Card_Arrival") == "card arrival"
    assert normalize("  Hello,   WORLD!!  ") == "hello world"
    assert normalize("top-up\tfailed\n") == "top up failed"
    assert normalize("___") == ""
    assert normalize(42) == "42"
    assert normalize("Card arrival") == normalize("card_arrival") == normalize("CARD-ARRIVAL.")


def test_label_index_follows_label_order():
    spec = TaskSpec(
        name="t",
        description="",
        modality="text",
        labels=[Label(name=n, definition="") for n in ("neg", "mid", "pos")],
    )
    assert spec.label_index == {"neg": 0, "mid": 1, "pos": 2}
    assert [spec.label_index[n] for n in spec.label_names] == [0, 1, 2]


def test_board_is_empty_until_written(tmp_path):
    p = Project("b", tmp_path).ensure()
    assert p.board_path == p.root / "board.json"
    assert not p.board_path.exists() and p.board() == {}
    p.board_path.write_text(json.dumps({"v1": {"builtin": {"objective": 0.5}}}))
    assert p.board() == {"v1": {"builtin": {"objective": 0.5}}}


def test_search_terms_drop_filler_words_and_keep_the_id():
    from byosm.data.hf_source.hub import search_terms

    terms = search_terms(["dair-ai/emotion dataset", "go-emojis benchmark"])
    assert terms[:2] == ["dair-ai/emotion dataset", "dair-ai/emotion"]
    assert "go-emojis" in terms and "dataset" not in terms and "benchmark" not in terms


def test_compiled_specs_keep_only_datasets_that_exist(monkeypatch):
    from byosm.data.hf_source import hub
    from byosm.spec import compiler

    monkeypatch.setattr(hub, "exists", lambda d: d == "real/one")
    kept = compiler._real([{"dataset": "real/one", "role": "labeled"}, {"dataset": "made/up", "role": "labeled"}])
    assert [c["dataset"] for c in kept] == ["real/one"]
