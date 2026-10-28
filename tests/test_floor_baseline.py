from binary_specs import binary

from byosm.predictors import ALWAYS_POSITIVE, BASELINES, get_baseline


def test_always_positive_answers_the_positive_label_whatever_the_input():
    spec = binary("t", "no", "yes")
    probs = get_baseline(ALWAYS_POSITIVE, spec).probs_batch(["a", "b"])
    assert probs[:, spec.pos_index].tolist() == [1.0, 1.0] and probs.sum() == 2
    assert BASELINES[ALWAYS_POSITIVE].params == 0
