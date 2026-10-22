from byosm.project import Project
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
    p = Project("b", tmp_path)
    p.save_spec(TaskSpec(name="b", description="b", modality="text", labels=[Label(name="a", definition="a")]))
    assert p.board() == {}
    p.save_experiment("v1", meta={})
    p.save_board(
        {"v1": {"builtin": {"kind": "experiment", "objective": 0.5}}, "gone": {"builtin": {"kind": "baseline"}}}
    )
    assert p.board() == {"v1": {"builtin": {"kind": "experiment", "objective": 0.5}}}


def test_search_terms_drop_filler_words_and_keep_the_id():
    from byosm.data.hf_source.hub import search_terms

    terms = search_terms(["dair-ai/emotion dataset", "go-emojis benchmark"])
    assert terms[:2] == ["dair-ai/emotion", "go-emojis"]
    assert "dataset" not in terms and "benchmark" not in terms


def test_datasets_the_spec_writer_names_become_search_terms_not_sources():
    from byosm.spec import compiler

    named = compiler._named([{"dataset": "facebook/multi_woz", "role": "labeled"}, {"dataset": "speech-ocean"}])
    assert named == ["multi woz", "speech ocean"]


def test_label_names_match_exactly_or_by_abbreviation():
    from byosm.text import match_label

    ours = ["world", "sports", "business", "science_technology", "other"]
    assert [match_label(t, ours) for t in ["World", "Sci/Tech", "neutral"]] == ["world", "science_technology", None]
    assert match_label("fear", ["fear", "fearful"]) == "fear"
    assert match_label("sci", ["science", "scifi"]) is None
