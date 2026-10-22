from byosm.ablation import _slug, _verdict


def _row(p_better, delta, test_delta):
    return {"calib": {"p_better": p_better, "delta": delta}, "test_delta": test_delta}


def test_better_needs_every_seed_sure_on_calib_and_ahead_on_test():
    assert _verdict([_row(0.97, 0.02, 0.01), _row(0.99, 0.03, 0.02)]) == "better"
    assert _verdict([_row(0.97, 0.02, 0.01), _row(0.90, 0.01, 0.02)]) == "not sure"
    assert _verdict([_row(0.97, 0.02, -0.01), _row(0.99, 0.03, 0.02)]) == "not sure"
    assert _verdict([_row(0.02, -0.03, -0.01), _row(0.01, -0.02, -0.02)]) == "worse"


def test_a_change_names_its_experiments():
    assert _slug({"train.pretrain_steps": 200}) == "pretrain_steps-200".replace("_", "-")
