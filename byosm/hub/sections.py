from __future__ import annotations

from .facts import CardFacts

SYNTH_LINE = "- LLM-written examples, double-labelled (generator + blind judge); held-out sets from an independent evaluator prompt\n"
STREAM_SECTION = """
### Streaming

```python
s = m.stream()
for chunk in mic_chunks():         # float32 mono, any size
    out = s.step(chunk)            # {"probs": [...per 10ms...], "speaking": bool, "events": [{"type": "start"|"end", "t_s": ...}]}
```
"""
EXAMPLES = {
    "audio": 'import soundfile as sf\nwav, sr = sf.read("clip.wav", dtype="float32")   # 16 kHz mono\nprint(m.predict(wav)["segments"])',
    "text": 'print(m.predict("so what I wanted to ask was"))',
    "image": 'print(m.predict("photo.jpg"))',
    "document": 'print(m.predict("invoice.pdf"))       # PDF or page image; scans are OCR\'d',
}
JEPA_STEP = (
    "3. **LeJEPA pretraining** ({steps} steps): multi-view prediction + SIGReg (isotropic-Gaussian\n"
    "   regulariser on random projections), no EMA teacher, no stop-gradient.\n"
)
TRAIN_STEPS = (
    "{n0}. **Supervised training**: {recipe}; checkpoint chosen on dev (best epoch {epoch}).\n"
    "{n1}. **Evaluation**: frozen eval suites; decision policy calibrated on each suite's calib split; baselines scored\n"
    "   identically; promotion decided on calib, test only reported.\n"
    "{n2}. **Improvement lineage**"
)


def labels(f: CardFacts) -> str:
    return "\n".join(f"- **`{lab.name}`**: {lab.definition}" for lab in f.spec.labels)


def _source_line(s) -> str:
    part = f", partition `{s.partition}`" if s.partition != "all" else ""
    return f"- [`{s.dataset}`](https://huggingface.co/datasets/{s.dataset}): role `{s.role}`" + part


def sources(f: CardFacts) -> str:
    return "\n".join(_source_line(s) for s in f.spec.data.hf_sources) or "- (none)"


def synth(f: CardFacts) -> str:
    return SYNTH_LINE if f.spec.data.synthetic else ""


def lineage_list(f: CardFacts) -> str:
    out = []
    for step in f.lineage:
        if not step["applied"]:
            out.append(f"- **{step['exp']}**: initial spec")
        for pr in step["applied"]:
            out.append(f"- **{step['exp']}** ← `{pr['id']}` (from {step['parent']}): {pr['why']}")
    return "\n".join(out)


def _in_domain_suites(f: CardFacts) -> list[str]:
    srcs = f.spec.data.hf_sources
    return [
        e.name
        for e in f.base.evals
        if e.source and any(s.dataset == e.source.dataset and s.partition == "train" for s in srcs)
    ]


def caveats(f: CardFacts) -> str:
    out = []
    in_domain = _in_domain_suites(f)
    if in_domain:
        out.append(
            f"- Trained on the **train partition** of the source behind {', '.join(in_domain)} (file-level hash split, "
            "disjoint from that suite's calib/test files). Results there are in-domain; baselines may not have seen that domain."
        )
    if f.nc:
        out.append(
            f"- Training data includes {', '.join(f.nc)}: **non-commercial**. Check licences before commercial use."
        )
    sizes = ", ".join(f"{s} ({r['n_test']} test items)" for s, r in f.results.items())
    out.append(f"- Evaluated on {sizes}. Small suites have wide confidence intervals.")
    return "\n".join(out)


def stream(f: CardFacts) -> str:
    return STREAM_SECTION if (f.p.root / "export" / "stream.onnx").exists() else ""


def example(f: CardFacts) -> str:
    return EXAMPLES[f.spec.modality]


def pip_packages(f: CardFacts) -> str:
    req = f.p.root / "export" / "requirements.txt"
    reqs = (
        [line.partition("#")[0].strip() for line in req.read_text().splitlines()]
        if req.exists()
        else ["numpy", "onnxruntime"]
    )
    return " ".join(["huggingface_hub"] + [r for r in reqs if r])


def built(f: CardFacts) -> str:
    if f.jepa:
        return "a JEPA-style encoder pretrained with LeJEPA and trained only for this task"
    if f.encoders:
        return f"fine-tuned from {' + '.join(f'[`{e}`](https://huggingface.co/{e})' for e in f.encoders)} for this task only"
    return "trained only for this task"


def notes(f: CardFacts) -> str:
    notes_f = f.p.root / "CARD.md"
    return f"\n## Notes\n\n{notes_f.read_text().strip()}\n" if notes_f.exists() else ""


def _precision_part(precision: str) -> str:
    if precision == "fp32":
        return ""
    return "mixed precision" if precision == "auto" else f"{precision} precision"


def recipe(f: CardFacts) -> str:
    tc = f.spec.train
    parts = [
        f"layer-wise LR decay on pretrained encoders (floor {tc.layer_decay_floor:g})" if f.encoders else "",
        "EMA of the weights" if tc.ema else "",
        _precision_part(tc.precision),
        "the benchmark's builtin test split folded into training" if f.meta.get("n_test_folded") else "",
    ]
    return ", ".join(x for x in parts if x)


def build_steps(f: CardFacts) -> str:
    jepa = JEPA_STEP.format(steps=f.spec.train.pretrain_steps) if f.jepa else ""
    n0 = 4 if f.jepa else 3
    rec = recipe(f) or "class-weighted cross-entropy"
    epoch = f.meta["best_epoch"] + 1
    return jepa + TRAIN_STEPS.format(n0=n0, n1=n0 + 1, n2=n0 + 2, recipe=rec, epoch=epoch)
