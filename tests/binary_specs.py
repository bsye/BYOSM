import numpy as np

from byosm.predictors import Predictor
from byosm.spec import Label, Objective, TaskSpec


def binary(name, negative, positive, metric="f1"):
    labels = [Label(name=negative, definition=""), Label(name=positive, definition="")]
    return TaskSpec(
        name=name,
        description="",
        modality="text",
        labels=labels,
        positive_label=positive,
        objective=Objective(metric=metric, constraint=0.98),
    )


class Scripted(Predictor):
    def __init__(self, spec, chance):
        super().__init__(spec)
        self.chance = chance

    def probs_batch(self, raws):
        p = np.array([self.chance(x) for x in raws], np.float32)
        out = np.zeros((len(raws), 2), np.float32)
        out[:, self.spec.pos_index], out[:, 1 - self.spec.pos_index] = p, 1 - p
        return out
