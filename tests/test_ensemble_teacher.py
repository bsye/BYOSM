import numpy as np

from byosm.project import Project
from byosm.spec import Label, TaskSpec
from byosm.train.teacher import averaged, teachers


def test_a_seeded_teacher_hands_down_the_average_of_up_to_three_of_its_runs(ws):
    p = Project("tc").ensure()
    p.save_spec(TaskSpec(name="tc", description="", modality="text", labels=[Label(name="a", definition="")]))
    for e in ("v3", "v3-s1", "v3-s2", "v3-s3", "v4"):
        p.save_experiment(e, meta={"params": 1})
    p.save_experiment("v3-s4", proposal={"applied": []})
    assert teachers(p, "v3", 3) == ["v3", "v3-s1", "v3-s2"]
    assert teachers(p, "v4", 3) == ["v4"] and teachers(p, "v3", 1) == ["v3"]
    assert teachers(p, "v3-soup", 2) == ["v3-soup", "v3"]


def test_soft_targets_are_averaged_row_by_row():
    a = [np.array([0.9, 0.1]), np.array([0.2, 0.8])]
    b = [np.array([0.7, 0.3]), np.array([0.4, 0.6])]
    out = averaged([a, b])
    assert np.allclose(out[0], [0.8, 0.2]) and np.allclose(out[1], [0.3, 0.7])
