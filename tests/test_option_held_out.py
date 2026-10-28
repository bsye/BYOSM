import json

import pytest
from option_pool import pool
from pydantic import ValidationError

from byosm.data.store import CALIB
from byosm.options.create import pair_plan
from byosm.options.pairs import table_rows
from byosm.spec import Label, PairPlan, TaskSpec

EVAL = (CALIB, "test")


def _names(rows):
    return {json.loads(r["option"])["name"] for r in rows}


def test_calib_and_test_offer_only_tools_that_training_never_sees():
    rows = list(table_rows(pool(), pair_plan(held_out=True)))
    seen = [r for r in rows if r["split"] not in EVAL]
    held = [r for r in rows if r["split"] in EVAL]
    assert {r["split"] for r in rows} == {"train", "dev", CALIB, "test"}
    assert _names(seen) and _names(held) and not _names(seen) & _names(held)
    assert not any(name in r["text"] for name in _names(seen) for r in held)


def test_a_held_out_layout_appears_in_calib_and_test_only():
    for plan in (pair_plan(held_out=True), PairPlan(held_out_layouts=["prose"])):
        rows = list(table_rows(pool(), plan))
        by_split = {s: {r["layout"] for r in rows if r["split"] == s} for s in ("train", "dev", *EVAL)}
        assert by_split["train"] == by_split["dev"] == {"marked"}
        assert by_split[CALIB] == by_split["test"] == {"marked", "prose"}


def test_without_a_held_out_plan_dev_is_left_for_calib_to_be_carved_from():
    rows = list(table_rows(pool(), PairPlan()))
    assert {r["split"] for r in rows} == {"train", "dev", "test"} and {r["layout"] for r in rows} == {"marked"}


def test_training_cannot_use_a_layout_held_out_for_calib_and_test():
    with pytest.raises(ValidationError):
        PairPlan(layouts=["marked", "prose"], held_out_layouts=["prose"])


def test_a_spec_without_a_pair_plan_keeps_its_data_key_and_a_plan_change_rebuilds_the_data():
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    assert "pairs" not in spec.data.model_dump(exclude_defaults=True)
    with_plan = spec.model_copy(update={"data": spec.data.model_copy(update={"pairs": PairPlan()})})
    layouts = with_plan.model_copy(deep=True)
    layouts.data.pairs.layouts = ["marked", "json"]
    assert len({spec.data_key(), with_plan.data_key(), layouts.data_key()}) == 3
