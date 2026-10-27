import numpy as np

from byosm.predictors.zero_shot import ZeroShotLLM
from byosm.spec import Label, TaskSpec


class Scrambled:
    name = "fake"

    def __init__(self):
        self.calls = []

    def json(self, system, prompt, schema, **kw):
        items = [b for b in prompt.split("\n\nItem ") if b.split(":", 1)[0].strip().isdigit()]
        self.calls.append(len(items))
        answers = [{"n": int(b.split(":", 1)[0]), "label": b.split(":", 1)[1].strip()} for b in items]
        if len(answers) > 1:
            answers = list(reversed(answers))[1:]
        return {"labels": answers}


def test_llm_rival_answers_land_on_their_own_rows_even_out_of_order_or_missing(monkeypatch):
    import byosm.llm

    fake = Scrambled()
    monkeypatch.setattr(byosm.llm, "get_llm", lambda role, cfg=None: fake)
    labels = [Label(name="fits", definition=""), Label(name="no_fit", definition="")]
    spec = TaskSpec(name="t", description="", modality="text", labels=labels)
    rows = [{"id": str(i), "input": "fits" if i % 3 else "no_fit"} for i in range(6)]
    probs = ZeroShotLLM(spec).probs_rows(rows, None)
    got = [spec.label_names[int(np.argmax(p))] for p in probs]
    assert got == [r["input"] for r in rows] and fake.calls == [6, 1]
