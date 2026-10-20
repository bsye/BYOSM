from byosm.modalities import get_modality
from byosm.spec import Label, TaskSpec, TrainCfg
from byosm.train import build_model, param_groups


def test_param_groups_layer_decay(tiny_bert):
    spec = TaskSpec(
        name="t",
        description="",
        modality="text",
        backbone=tiny_bert,
        labels=[Label(name="a", definition=""), Label(name="b", definition="")],
        train=TrainCfg(lr_backbone=1e-4, lr_head=1e-3, layer_decay_floor=0.1),
    )
    model = build_model(spec, get_modality(spec))
    lr = {id(p): g["lr"] for g in param_groups(model, spec.train) for p in g["params"]}
    named = dict(model.backbone.named_parameters())
    emb = lr[id(named["model.embeddings.word_embeddings.weight"])]
    l0 = lr[id(named["model.encoder.layer.0.attention.self.query.weight"])]
    l3 = lr[id(named["model.encoder.layer.3.output.dense.weight"])]
    assert abs(emb - 1e-5) < 1e-12 and emb < l0 < l3 < 1e-4
    assert all(lr[id(p)] == 1e-3 for h in model.heads.values() for p in h.parameters())
    head_params = sum(1 for h in model.heads.values() for _ in h.parameters())
    assert len(lr) == sum(1 for _ in model.backbone.parameters()) + head_params
