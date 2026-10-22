import random

import pytest
from PIL import Image

from byosm.data.store import Dataset, record
from byosm.export import export
from byosm.modalities.document import OCRStore
from byosm.pipeline import build
from byosm.project import Project
from byosm.spec import DocumentCfg, EvalSuite, Label, TaskSpec, TrainCfg


def quiet(_):
    return None


def doc_spec(text_backbone):
    spec = TaskSpec(
        name="doc",
        description="invoice vs letter",
        modality="document",
        backbone="doc",
        labels=[Label(name="invoice", definition=""), Label(name="letter", definition="")],
        document=DocumentCfg(
            route="fused",
            text_backbone=text_backbone,
            image_backbone="tiny-cnn",
            max_pages=3,
            image_size=64,
            dim=64,
            ocr="never",
        ),
        train=TrainCfg(epochs=3, pretrain_steps=5, batch_size=8, lr_backbone=1e-4, pool="attention"),
        evals=[EvalSuite(name="builtin", primary=True)],
    )
    spec.data.synthetic = False
    return spec


def pages_for(label, i, rng):
    if label == "invoice":
        pages = [(f"INVOICE #{i}", ["Qty  Item  Price", f"Total due: ${rng.randint(10, 999)}"])]
    else:
        pages = [("Dear Sir or Madam,", ["I am writing to you about", "Kind regards"])]
    return pages + [("Appendix", ["terms and conditions"])] * rng.randint(0, 2)


def write_corpus(dd, make_pdf):
    (dd / "pdf").mkdir(parents=True)
    rng = random.Random(0)
    splits = {"train": [], "dev": [], "test": []}
    for i in range(60):
        lab = "invoice" if i % 2 else "letter"
        make_pdf(dd / "pdf" / f"{i}.pdf", pages_for(lab, i, rng))
        split = "train" if i < 40 else ("dev" if i < 50 else "test")
        splits[split].append(record(f"pdf/{i}.pdf", lab, "test"))
    for s, rows in splits.items():
        Dataset(dd).save(s, rows)


@pytest.mark.slow
def test_document_fused_pipeline_and_export(ws, tiny_bert, load_runtime, make_pdf):
    p = Project("doc").ensure()
    spec = doc_spec(tiny_bert)
    p.save_spec(spec)
    dd = p.data_dir_for(spec)
    write_corpus(dd, make_pdf)
    build("doc", log=quiet)
    assert p.board()["v1"]["builtin"]["objective"] > 0.8

    m = load_runtime(export("doc", log=quiet))
    pred = m.predict(str(dd / "pdf" / "45.pdf"))
    assert pred["label"] in ("invoice", "letter") and m.encode(str(dd / "pdf" / "45.pdf")).shape == (64,)


def test_ocr_store_key_follows_pixels(tmp_path):
    store = OCRStore(tmp_path)
    white = Image.new("RGB", (8, 8), "white")
    same = Image.new("RGB", (8, 8), "white")
    dark = Image.new("RGB", (8, 8), "white")
    dark.putpixel((3, 3), (0, 0, 0))
    assert store.key(white) == store.key(same)
    assert store.key(white) != store.key(dark)
    assert store.key(white) != store.key(white.convert("L"))
    assert store.key(white) != store.key(Image.new("RGB", (4, 16), "white"))


def test_ocr_store_round_trip(tmp_path):
    store = OCRStore(tmp_path)
    key = store.key(Image.new("RGB", (8, 8), "white"))
    assert store.get(key) is None
    store.put(key, "hello")
    assert store.get(key) == "hello" and (tmp_path / key[:2] / f"{key}.txt").exists()
