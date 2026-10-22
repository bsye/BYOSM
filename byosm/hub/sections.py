from __future__ import annotations

from ..data.budget import fit_line, load_fit
from ..data.store import TRAIN_INPUTS
from .facts import HUB_SOURCE, CardFacts

CARD_NOTES = "card-notes"

LLM_SOURCE = "llm:"
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


def _source_line(source: str, n: int, what: str) -> str:
    if source.startswith(HUB_SOURCE):
        name = source.removeprefix(HUB_SOURCE)
        return f"- [`{name}`](https://huggingface.co/datasets/{name}): {n:,} {what} rows"
    if source.startswith(LLM_SOURCE):
        return f"- {n:,} {what} rows written by `{source.removeprefix(LLM_SOURCE)}`"
    if source == TRAIN_INPUTS:
        return f"- {n:,} {what} rows: the training inputs again, for pretraining"
    return f"- {n:,} {what} rows from `{source}`"


def sources(f: CardFacts) -> str:
    lines = [
        _source_line(src, n, what)
        for split, what in (("train", "training"), ("unlabeled", "unlabeled"))
        for src, n in sorted(f.used.get(split, {}).items(), key=lambda kv: -kv[1])
    ]
    return "\n".join(lines) or "- (none)"


def _shrunk(f: CardFacts) -> str:
    fit = load_fit(f.p.data_dir_for(f.spec))
    if not fit or not fit["shrunk"]:
        return ""
    line = fit_line(fit["seconds_per_call"], fit["written"], fit["planned"], fit["budget_hours"] * 3600)
    return f"- Shrunk to the time budget: {line}\n"


def synth(f: CardFacts) -> str:
    return SYNTH_LINE + _shrunk(f) if f.spec.data.synthetic else ""


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
    from .provenance import example_input

    real = example_input(f)
    return f"print(m.predict({real!r}))" if real else EXAMPLES[f.spec.modality]


def pip_packages(f: CardFacts) -> str:
    req = f.p.root / "export" / "requirements.txt"
    reqs = (
        [line.partition("#")[0].strip() for line in req.read_text().splitlines()]
        if req.exists()
        else ["numpy", "onnxruntime"]
    )
    return " ".join(["huggingface_hub"] + [r for r in reqs if r])


def notes(f: CardFacts) -> str:
    text = f.p.report(CARD_NOTES)
    return f"\n## Notes\n\n{text.strip()}\n" if text else ""


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
