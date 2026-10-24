from byosm.app.readiness import readiness


def test_a_specialist_is_ready_only_when_its_interval_clears_the_strongest_rival():
    rival = {"name": "dit", "score": 0.94}
    assert readiness(0.948, rival, (0.941, 0.956)) == "ready"
    assert readiness(0.882, {"name": "silero", "score": 0.865}, (0.863, 0.899)) == "ties"
    assert readiness(0.797, {"name": "jev", "score": 0.807}, (0.78, 0.81)) == "behind"
    assert readiness(0.95, rival, None) == "no_interval"
    assert readiness(None, rival, None) == "unbuilt"


def test_a_rival_that_scores_zero_or_is_missing_is_not_something_beaten():
    assert readiness(0.895, {"name": "llm", "score": 0.0}, (0.86, 0.92)) == "no_rival"
    assert readiness(0.9, None, (0.8, 0.95)) == "no_rival"


def test_lower_is_better_scores_compare_the_rival_with_our_score_on_the_same_datasets():
    rival = {"name": "Timestral-209M", "score": 0.67, "ours": 0.74}
    assert readiness(0.70, rival, (0.6, 0.8), lower_is_better=True) == "behind"
    assert readiness(0.60, {"name": "t", "score": 0.67}, (0.55, 0.65), lower_is_better=True) == "ready"
