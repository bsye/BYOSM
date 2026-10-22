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


def test_summaries_average_the_seeds_and_arms_are_flagged(ws):
    from byosm.ablation import PREFIX, is_arm, summaries
    from byosm.project import Project
    from byosm.spec import Label, TaskSpec

    p = Project("q", ws)
    p.save_spec(TaskSpec(name="q", description="q", modality="text", labels=[Label(name="a", definition="")]))
    seed = {
        "seed": 1,
        "off": "abl-x-off-s1",
        "on": "abl-x-on-s1",
        "calib": {"delta": 0.02, "p_better": 0.9},
        "test": None,
    }
    report = {"base": "v1", "change": {"x": 1}, "verdict": "not sure", "seeds": [{**seed, "test_delta": 0.01}]}
    p.save_report(PREFIX + "x-1", report)
    p.save_report("other", {})
    [s] = summaries(p)
    assert (s["calib"], s["test"], s["verdict"]) == (0.02, 0.01, "not sure")
    assert s["seeds"] == [{"seed": 1, "calib": 0.02, "p_better": 0.9, "test": 0.01}]
    assert is_arm(seed["on"]) and not is_arm("v1")
