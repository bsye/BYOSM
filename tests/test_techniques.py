import pytest

from byosm.distill import TINY_TEXT as TINY
from byosm.improve.analyst import _prompt
from byosm.improve.paths import allowed, apply_ops
from byosm.improve.techniques import OTHER, eligible, family, techniques
from byosm.spec import Label, PairPlan, TaskSpec
from byosm.spec.backbones import EMBEDDING_MODEL, MINILM, MULTILINGUAL_EMBEDDING_MODEL

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
    "train.pairs_per_row",
    "train.lr_pairs",
    "train.precision",
    "document.route",
    "document.image_backbone",
    "predict.rates",
    "predict.routing",
    "data.pairs.layouts",
    "data.pairs.mask_names",
    "data.pairs.paraphrases",
    "data.pairs.near_misses",
    "fit.pretraining",
    "fit.quantile_boosting",
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


def _spec(modality="text", **fields):
    labels = [Label(name=n, definition="") for n in "ab"]
    return TaskSpec(name="t", description="", modality=modality, labels=labels, **fields)


def _method(value):
    return {"op": "set", "path": "train.method", "value": value}


def test_every_editable_path_belongs_to_one_available_family_and_seeds_are_left_to_replicas():
    for path in EDITABLE:
        assert allowed(path) and family([{"path": path}]) != OTHER, path
    assert not allowed("train.seed")


def test_every_ranked_technique_has_an_entry_and_a_planned_one_adds_no_editable_path():
    ids = {t.id for t in techniques()}
    assert set(RANKED) <= ids and len(ids) == len(techniques())
    assert all(not t.paths for t in techniques() if t.status != "available")


def test_a_family_is_offered_only_to_the_kinds_it_fits():
    text = {t.id for t in eligible(_spec())}
    audio = {t.id for t in eligible(_spec("audio"))}
    tagger = {t.id for t in eligible(_spec(task_type="frame"))}
    assert {"targeted-data", "embedding-head", "setfit"} <= text and "audio-mix" not in text
    assert "audio-mix" in audio and "targeted-data" not in audio
    assert "targeted-data" in tagger and not {"embedding-head", "setfit"} & tagger


def test_a_path_pinned_to_a_value_tags_only_edits_that_set_that_value():
    assert allowed("train.method")
    assert family([_method("embedding-head")]) == "embedding-head"
    assert family([_method("setfit"), {"op": "set", "path": "train.pairs_per_row", "value": 20}]) == "setfit"
    assert family([_method("finetune")]) == OTHER


@pytest.mark.parametrize(
    ("fields", "ops", "backbone"),
    [
        ({"backbone": TINY}, [_method("setfit")], EMBEDDING_MODEL),
        ({"backbone": TINY, "languages": ["en", "it"]}, [_method("embedding-head")], MULTILINGUAL_EMBEDDING_MODEL),
        ({"backbone": TINY}, [_method("setfit"), {"op": "set", "path": "backbone", "value": TINY}], TINY),
        ({"backbone": MULTILINGUAL_EMBEDDING_MODEL}, [_method("setfit")], MULTILINGUAL_EMBEDDING_MODEL),
        ({"backbone": TINY}, [{"op": "set", "path": "train.epochs", "value": 3}], TINY),
        ({"backbone": MINILM}, [_method("finetune")], MINILM),
    ],
)
def test_switching_to_an_embedding_method_picks_a_sentence_encoder_unless_one_is_named(fields, ops, backbone):
    assert apply_ops(_spec(**fields), ops).backbone == backbone


def test_editing_an_embedding_experiment_keeps_its_encoder():
    spec = apply_ops(
        _spec(backbone=TINY), [_method("embedding-head"), {"op": "set", "path": "backbone", "value": TINY}]
    )
    assert apply_ops(spec, [{"op": "set", "path": "train.seed", "value": 1}]).backbone == TINY


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
    assert '"train.method=setfit"' in prompt and "=v means set that path to exactly v" in prompt


PAIR_FAMILIES = {"input-layouts", "near-miss-negatives"}


def test_pair_families_are_offered_only_to_a_specialist_with_a_pair_plan_and_their_edits_make_a_valid_spec():
    plain, pairs = _spec(), _spec()
    pairs.data.pairs = PairPlan()
    assert not PAIR_FAMILIES & {t.id for t in eligible(plain)} and {t.id for t in eligible(pairs)} >= PAIR_FAMILIES
    ops = [
        {"op": "set", "path": "data.pairs.layouts", "value": ["marked", "json", "openai"]},
        {"op": "set", "path": "data.pairs.mask_names", "value": 0.3},
        {"op": "set", "path": "data.pairs.paraphrases", "value": 2},
        {"op": "set", "path": "data.pairs.near_misses", "value": ["nearest", "removed"]},
    ]
    assert [family([o]) for o in ops] == ["input-layouts"] * 3 + ["near-miss-negatives"]
    edited = apply_ops(pairs, ops).data.pairs
    assert (edited.layouts, edited.mask_names, edited.paraphrases) == (["marked", "json", "openai"], 0.3, 2)
    assert edited.near_misses == ["nearest", "removed"]
