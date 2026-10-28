from byosm.improve.keys import proposal_key


def _key(*ops):
    return proposal_key({"ops": [{"op": "set", "path": p, "value": v} for p, v in ops]})


def test_a_nearby_value_or_a_reordered_edit_counts_as_the_same_experiment():
    assert _key(("train.lr_backbone", 2e-5), ("train.epochs", 8)) == _key(
        ("train.epochs", 8), ("train.lr_backbone", 2.2e-5)
    )
    assert _key(("hard_cases", "Refund  Requests ")) == _key(("hard_cases", "refund requests"))


def test_a_different_order_of_magnitude_or_another_seed_is_a_new_experiment():
    assert _key(("train.lr_backbone", 2e-5)) != _key(("train.lr_backbone", 1e-4))
    assert _key(("train.seed", 1)) != _key(("train.seed", 2))
    assert _key(("train.freeze_backbone", True)) != _key(("train.freeze_backbone", 1))
