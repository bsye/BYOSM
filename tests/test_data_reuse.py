from byosm.data.budget import SYNTH_KEY_FILE, reuse_written
from byosm.data.store import Dataset, record
from byosm.spec import Label, TaskSpec


def _spec(**data):
    spec = TaskSpec(
        name="t",
        description="route a message",
        modality="text",
        labels=[Label(name="a", definition="x"), Label(name="b", definition="y")],
    )
    spec.data.train_size, spec.data.test_size = 10, 4
    for k, v in data.items():
        setattr(spec.data, k, v)
    return spec


def test_rows_the_llm_wrote_for_another_data_plan_are_reused_when_they_would_be_written_the_same(tmp_path):
    spec = _spec()
    old = tmp_path / "old"
    Dataset(old).save(
        "train", [record(f"written {i}", "a", "llm:model") for i in range(6)] + [record("real", "b", "hf:x")]
    )
    (old / SYNTH_KEY_FILE).write_text(spec.synth_key())
    other = _spec(hf_queries=["something else"])
    assert other.data_key() != spec.data_key() and other.synth_key() == spec.synth_key()
    splits = {"train": [], "dev": [], "test": []}
    reuse_written(other, splits, tmp_path / "new", log=lambda m: None)
    assert [r["input"] for r in splits["train"]] == [f"written {i}" for i in range(6)]
    assert (tmp_path / "new" / SYNTH_KEY_FILE).read_text() == spec.synth_key()


def test_rows_written_for_different_labels_are_not_reused(tmp_path):
    old = tmp_path / "old"
    Dataset(old).save("train", [record("written", "a", "llm:model")])
    (old / SYNTH_KEY_FILE).write_text("someotherkey")
    splits = {"train": [], "dev": [], "test": []}
    reuse_written(_spec(), splits, tmp_path / "new", log=lambda m: None)
    assert splits["train"] == []


def test_the_input_length_grows_to_cover_the_training_rows(tmp_path, monkeypatch):
    import transformers

    from byosm.train.input_length import fitted_max_len

    class Tok:
        model_max_length = 8192

        def __call__(self, text):
            return {"input_ids": text.split()}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda repo: Tok())
    spec = _spec()
    spec.backbone = "hf:any/model"
    Dataset(tmp_path).save("train", [record(" ".join(["w"] * n), "a", "llm:m") for n in (50, 100, 300, 300, 300)])
    assert fitted_max_len(spec, tmp_path, log=lambda m: None).train.max_len == 320
