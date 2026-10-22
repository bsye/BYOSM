import transformers

from byosm.predictors import hf
from byosm.spec import Label, TaskSpec


class _Config:
    id2label = {0: "LABEL_0", 1: "LABEL_1"}


class _Model:
    def parameters(self):
        return []


def test_a_model_with_generic_labels_can_be_mapped_onto_ours_by_hand(monkeypatch):
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda repo: _Config())
    monkeypatch.setattr(hf, "load_model", lambda repo, visual: (None, _Model()))
    labels = [Label(name="finished", definition="done"), Label(name="unfinished", definition="not done")]
    spec = TaskSpec(name="t", description="turn", modality="text", labels=labels)
    rival = hf.HFClassifier(spec, "org/turns#LABEL_1=finished,LABEL_0=unfinished")
    assert rival.id == "hf:org/turns#LABEL_1=finished,LABEL_0=unfinished"
    assert rival.map == {1: 0, 0: 1} and rival.missing == []
