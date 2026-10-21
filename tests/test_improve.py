import pytest

from byosm.data import hf_source
from byosm.improve import apply_ops, rule_proposals
from byosm.spec import DocumentCfg, HFSource, Label, TaskSpec, TrainCfg

RISING = [0.6, 0.7, 0.75, 0.78, 0.79, 0.8, 0.82, 0.83]
PLATEAU = [0.9, 0.95, 0.96, 0.96]
OVERFIT_CURVE = [0.7, 0.8, 0.79, 0.78, 0.78]
OVERFIT_LOSSES = [1.0, 0.6, 0.4, 0.3, 0.2]
REQUESTED = 100
CAPPED_AVAILABLE = 150
SHORT_AVAILABLE = 110
EXHAUSTED_REQUEST = 101


def diagnosis(spec, objective, baseline, curve, best_epoch, losses=None):
    suite = {
        "objective": objective,
        "best_baseline": ("hf:x", baseline) if baseline else None,
        "threshold": None,
        "confusions": [("form", "specification", 5)],
        "weakest": [],
    }
    return {
        "spec": spec,
        "base": spec,
        "primary": "builtin",
        "n_train": 100,
        "dev": curve[best_epoch],
        "binary": False,
        "curve": curve,
        "losses": losses or [1.0] * len(curve),
        "train_seconds": 60,
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "epochs_cfg": spec.train.epochs,
        "late_best": best_epoch >= int(0.75 * len(curve)) and len(curve) >= spec.train.epochs,
        "suites": {"builtin": suite},
        "board": {},
    }


def proposals(spec, *args):
    return {p["id"]: p for p in rule_proposals(diagnosis(spec, *args))}


def document_spec():
    spec = TaskSpec(
        name="d",
        description="",
        modality="document",
        labels=[Label(name="form", definition=""), Label(name="specification", definition="")],
        document=DocumentCfg(image_backbone="hf:facebook/dinov2-small"),
        train=TrainCfg(epochs=8),
    )
    spec.data.synthetic = False
    source = HFSource(
        dataset="x/y",
        role="labeled",
        sampling="balanced",
        per_label=REQUESTED,
        input_column="image",
        label_column="label",
    )
    spec.data.hf_sources = [source]
    return spec


@pytest.fixture
def limited_split(monkeypatch):
    def availability(src, sp):
        form = CAPPED_AVAILABLE if src.per_label == REQUESTED else SHORT_AVAILABLE
        return {"form": form, "specification": 900}

    monkeypatch.setattr(hf_source, "availability", availability)
    monkeypatch.setattr(hf_source, "split_sizes", lambda ds, cfg=None: {"train": 50000, "val": 300})
    spec = document_spec()
    spec.objective.target = 0.95
    return spec


def test_losing_to_baseline_while_rising_proposes_data_encoder_and_longer_training():
    spec = document_spec()
    ids = proposals(spec, 0.80, 0.95, RISING, 7)
    assert {"more-data", "domain-encoder", "train-longer"} <= set(ids)
    assert "still rising" in ids["more-data"]["why"]
    new = apply_ops(spec, [o for k in ("more-data", "domain-encoder") for o in ids[k]["ops"]])
    assert new.data.hf_sources[0].per_label == 400 and new.document.image_backbone == "hf:microsoft/dit-base"


def test_at_target_and_plateaued_proposes_nothing():
    spec = document_spec()
    spec.objective.target = 0.8
    assert rule_proposals(diagnosis(spec, 0.97, 0.95, PLATEAU, 2)) == []


def test_more_data_is_capped_at_what_the_split_holds(limited_split):
    ids = proposals(limited_split, 0.80, 0.95, RISING, 7)
    assert ids["more-data"]["ops"][0]["value"] == CAPPED_AVAILABLE


def test_exhausted_source_proposes_new_split(limited_split):
    limited_split.data.hf_sources[0].per_label = EXHAUSTED_REQUEST
    assert SHORT_AVAILABLE < 1.5 * EXHAUSTED_REQUEST
    ids = proposals(limited_split, 0.80, 0.95, RISING, 7)
    assert ids["new-split"]["ops"][0]["value"]["split"] in ("train", "val") and "more-data" not in ids


def test_overfitting_proposes_regularize(limited_split):
    limited_split.data.hf_sources[0].per_label = EXHAUSTED_REQUEST
    ids = set(proposals(limited_split, 0.80, 0.95, OVERFIT_CURVE, 1, OVERFIT_LOSSES))
    assert "regularize" in ids


def test_more_data_grows_the_labeled_source_that_supplied_rows():
    from byosm.improve.rules_data import _richest_labeled

    spec = document_spec()
    spec.data.hf_sources = [HFSource(dataset="broken"), HFSource(dataset="org/real"), HFSource(dataset="org/small")]
    dg = diagnosis(spec, 0.8, 0.9, RISING, 7)
    dg["rows_by_source"] = {"hf:org/real": 3000, "hf:org/small": 10, "llm:x": 2000}
    assert _richest_labeled(dg)[0] == 1
    dg["rows_by_source"] = {}
    assert _richest_labeled(dg)[0] == 0


def test_more_data_is_proposed_for_any_gap_to_a_baseline():
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    spec.objective.target = 0.5
    assert "more-data" in proposals(spec, 0.92, 0.93, PLATEAU, 2)


def test_streamed_source_read_to_the_end_proposes_an_unused_split(monkeypatch):
    monkeypatch.setattr(hf_source, "split_sizes", lambda ds, cfg=None: {"train": 50, "validation": 30})
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    spec.data.hf_sources = [HFSource(dataset="org/d", role="labeled", split="train", max_rows=80)]
    ids = proposals(spec, 0.8, 0.9, RISING, 7)
    value = ids["new-split"]["ops"][0]["value"]
    assert (value["split"], value["max_rows"]) == ("validation", 30) and "more-data" not in ids
