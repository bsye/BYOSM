import numpy as np
from binary_specs import Scripted, binary

from byosm.evals import cutoffs
from byosm.evals.grouped import summary
from byosm.evals.scoring import score
from byosm.predictors import ALWAYS_POSITIVE, get_baseline


def span_rows(turn, durations, chances, split="test"):
    rows = []
    for i, (duration, states) in enumerate(zip(durations, chances, strict=True)):
        label = "finished" if i == len(durations) - 1 else "speaking"
        for k, (at, p) in enumerate(states):
            meta = {"span": f"{turn}:{i}", "span_s": duration, "at_s": at, "turn": turn}
            rows.append({"id": f"{turn}:{i}:{k}", "input": f"{p}", "label": label, "meta": meta, "split": split})
    return rows


def turn_spec():
    return binary("turn", "speaking", "finished", "recall_at_precision")


def test_always_finished_reproduces_a_plain_silence_timeout_at_each_latency_budget(tmp_path):
    spec = turn_spec()
    test = span_rows("a", [0.25, 0.45, 0.7, 1.2], [[(0, 1)]] * 4) + span_rows("b", [0.35, 0.8], [[(0, 1)]] * 2)
    calib = span_rows("c", [0.5, 0.9], [[(0, 1)]] * 2, "calib")
    m = score(get_baseline(ALWAYS_POSITIVE, spec), spec, calib, test, tmp_path)["metrics"]["cutoffs"]
    holds = np.array([0.25, 0.45, 0.7, 0.35, 0.5])
    assert m["hold_spans"] == 5 and m["eot_spans"] == 3
    for budget in cutoffs.LATENCY_BUDGETS_MS:
        assert m["all"][str(budget)]["cutoff"] == (holds > budget / 1000).mean()
        assert m["all"][str(budget)]["latency"] <= budget / 1000 + 1e-9


def test_a_detector_that_tells_pauses_from_ends_cuts_nobody_off_and_answers_fast(tmp_path):
    spec = turn_spec()
    chances = [[(0, 0.1)], [(0, 0.1)], [(0, 0.045), (0.4, 0.9)]]
    test, calib = span_rows("a", [0.5, 0.9, 1.5], chances), span_rows("b", [0.6, 1.1, 1.4], chances, "calib")
    m = score(Scripted(spec, float), spec, calib, test, tmp_path)["metrics"]["cutoffs"]
    assert m["all"]["300"] == {"threshold": 0.04, "action_delay": 0.2, "timeout": 1.0, "cutoff": 1.0, "latency": 0.2}
    assert m["all"]["600"] == {"threshold": 0.89, "action_delay": 0.2, "timeout": 1.5, "cutoff": 0.0, "latency": 0.4}
    assert m["held_out"]["300"]["cutoff"] == 1.0 and m["held_out"]["600"]["cutoff"] == 0.0
    assert summary({"cutoffs": m}) == "cutoffs 100.0% at 300 ms, 0.0% at 600 ms"


def test_a_budget_no_policy_meets_is_shown_as_such():
    held = {"300": None, "600": {"cutoff": 0.05}}
    assert summary({"cutoffs": {"held_out": held}}) == "cutoffs no policy that fast at 300 ms, 5.0% at 600 ms"


def test_a_span_fires_at_the_first_state_whose_chance_passes_the_threshold():
    rows = span_rows("a", [0.9, 1.0], [[(0, 0.2), (0.3, 0.7)], [(0, 0.9)]])
    s = cutoffs.spans(turn_spec(), rows, np.array([0.2, 0.7, 0.9]))
    at = {t: s.fire[i] for i, t in enumerate(cutoffs.THRESHOLDS)}
    assert at[0.1].tolist() == [0, 0] and at[0.5].tolist() == [0.3, 0] and np.isinf(at[0.8][0])


def test_a_suite_of_ends_only_reports_no_cutoffs(tmp_path):
    spec = turn_spec()
    rows = span_rows("a", [0.9], [[(0, 1)]])
    assert score(get_baseline(ALWAYS_POSITIVE, spec), spec, [], rows, tmp_path)["metrics"]["cutoffs"] == {}
