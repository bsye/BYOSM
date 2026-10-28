from byosm.improve.analyst import _prompt
from byosm.improve.paths import allowed
from byosm.improve.techniques import OTHER, eligible, family, techniques
from byosm.spec import Label, TaskSpec

EDITABLE = [
    "data.hf_sources",
    "data.hf_sources.0.per_label",
    "data.hf_queries",
    "data.train_size",
    "data.audio.fill_ms",
    "data.synthetic",
    "hard_cases",
    "diversity_axes",
    "backbone",
    "train.epochs",
    "train.lr_backbone",
    "train.pretrain_steps",
    "train.refine_rounds",
    "train.freeze_backbone",
    "train.layer_decay_floor",
    "train.precision",
    "document.route",
    "document.image_backbone",
]
RANKED = [
    "multi-rate-forecast",
    "forecast-routing",
    "input-layouts",
    "near-miss-negatives",
    "embedding-head",
    "setfit",
    "label-noise",
    "weight-averaging",
    "learning-curve",
    "forecaster-pretraining",
    "reinit-top",
    "small-search",
    "consistency",
    "quantile-boosting",
    "self-training",
]


def _spec(modality="text"):
    return TaskSpec(name="t", description="", modality=modality, labels=[Label(name=n, definition="") for n in "ab"])


def test_every_editable_path_belongs_to_one_available_family_and_seeds_are_left_to_replicas():
    for path in EDITABLE:
        assert allowed(path) and family([{"path": path}]) != OTHER, path
    assert not allowed("train.seed")


def test_every_ranked_technique_has_an_entry_and_a_planned_one_adds_no_editable_path():
    ids = {t.id for t in techniques()}
    assert set(RANKED) <= ids and len(ids) == len(techniques())
    assert all(not t.paths for t in techniques() if t.status == "planned")


def test_a_family_is_offered_only_to_the_kinds_it_fits():
    text = {t.id for t in eligible(_spec())}
    audio = {t.id for t in eligible(_spec("audio"))}
    assert "targeted-data" in text and "audio-mix" not in text
    assert "audio-mix" in audio and "targeted-data" not in audio


def test_the_proposer_reads_the_catalogue_with_how_often_each_family_was_tried():
    dg = {
        "spec": _spec(),
        "board": {},
        "curve": [],
        "suites": {},
        "history": [{"experiment": "v2", "family": "schedule"}, {"experiment": "v3", "family": "schedule"}],
    }
    prompt = _prompt(dg)
    assert '"family": "schedule"' in prompt and '"tried": 2' in prompt
    assert "audio-mix" not in prompt and "multi-rate-forecast" not in prompt
