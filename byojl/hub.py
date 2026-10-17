"""Publish a specialist to the Hugging Face Hub: exported bundle + generated model card.

The card is written from the project's own artifacts, so it can't drift from what was measured:
  spec.yaml   -> what the model does, labels and definitions, objective, data sources
  board.json  -> test results on every eval suite, side by side with every baseline (also as `model-index` metadata)
  experiments -> lineage: which `improve` proposals produced the shipped model, and the evidence for each
  manifest    -> size, decision policy, preprocessing, usage
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import Project
from .spec import TaskSpec

PIPELINE_TAG = {("audio", "frame"): "voice-activity-detection", ("audio", "classify"): "audio-classification",
                ("text", "classify"): "text-classification", ("text", "frame"): "token-classification", ("image", "classify"): "image-classification"}
TASK_NAME = {"voice-activity-detection": "Voice Activity Detection", "audio-classification": "Audio Classification",
             "text-classification": "Text Classification", "token-classification": "Token Classification", "image-classification": "Image Classification"}
NC_DATASETS = {"ashraq/esc50": "ESC-50 (CC BY-NC 3.0)"}


def _f(x: Any, nd: int = 4) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _lineage(p: Project, exp: str) -> list[dict]:
    chain, cur = [], exp
    while cur:
        f = p.exp_dir(cur) / "proposal.json"
        if not f.exists():
            chain.append({"exp": cur, "applied": []})
            break
        pr = json.loads(f.read_text())
        chain.append({"exp": cur, "parent": pr["parent"], "applied": pr["applied"]})
        cur = pr["parent"]
    return list(reversed(chain))


def _suite_dataset(spec: TaskSpec, suite: str) -> tuple[str, str]:
    s = next(e for e in spec.evals if e.name == suite)
    if s.kind == "hf" and s.source:
        return s.source.dataset, s.source.dataset
    return f"{spec.name}-{suite}", f"byojl/{spec.name}-{suite}"


def model_card(name: str, repo_id: str, license: str) -> str:
    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    m = json.loads((p.root / "export" / "manifest.json").read_text())
    exp = m["experiment"]
    spec = TaskSpec.load(p.exp_dir(exp) / "spec.yaml")
    meta = json.loads((p.exp_dir(exp) / "meta.json").read_text())
    board = json.loads((p.root / "board.json").read_text())
    tag = PIPELINE_TAG.get((spec.modality, spec.task_type), "other")
    prim = base.primary_suite()
    datasets = sorted({s.dataset for s in spec.data.hf_sources} | {e.source.dataset for e in base.evals if e.source})
    nc = [NC_DATASETS[d] for d in datasets if d in NC_DATASETS]

    # ---- metadata (model-index renders as an evaluation widget on the Hub)
    results = []
    for suite, r in board.get(exp, {}).items():
        ds_name, ds_type = _suite_dataset(base, suite)
        mm = r["metrics"]
        metrics = [{"type": base.objective.metric, "value": round(r["objective"], 4), "name": f"{base.objective.metric} ({base.decision.kind} decision, calibrated)"}]
        metrics += [{"type": k, "value": round(mm[k], 4), "name": k} for k in ("precision", "recall", "roc_auc", "avg_precision") if mm.get(k) is not None]
        results.append({"task": {"type": tag, "name": TASK_NAME.get(tag, tag)}, "dataset": {"name": ds_name, "type": ds_type, "split": "test" if suite != "builtin" else "byojl-heldout"}, "metrics": metrics})
    front = {"license": license, "library_name": "onnx", "pipeline_tag": tag, "tags": ["byojl", "jepa", "lejepa", "onnx", "system-1", spec.modality, spec.backbone],
             "datasets": [d for d in datasets], "model-index": [{"name": repo_id.split("/")[-1], "results": results}]}

    # ---- comparison table: every predictor on every suite
    suites = [e.name for e in base.evals]
    rows = sorted(board.items(), key=lambda kv: -(kv[1].get(prim, {}).get("objective") or -1))
    head = "| model | params | " + " | ".join(f"{s} {base.objective.metric}" + (" (primary)" if s == prim else "") for s in suites) + " | " + " | ".join(f"{s} ROC-AUC" for s in suites) + " |"
    sep = "|" + "---|" * (2 + 2 * len(suites))
    lines = [head, sep]
    for pid, by in rows:
        kind = next(iter(by.values()))["kind"]
        label = f"**{pid} (this model)**" if pid == exp else (f"{pid} (BYOJL experiment)" if kind == "experiment" else f"{pid} (baseline)")
        params = next(iter(by.values())).get("params")
        cells = [label, f"{params / 1e6:.2f}M" if params else "–"]
        cells += [f"**{_f(by[s]['objective'])}**" if pid == exp and s in by else _f(by.get(s, {}).get("objective")) for s in suites]
        cells += [_f(by.get(s, {}).get("metrics", {}).get("roc_auc"), 3) for s in suites]
        lines.append("| " + " | ".join(cells) + " |")
    comparison = "\n".join(lines)

    # ---- detailed test eval for this model
    det = ["| suite | dataset | test items | " + base.objective.metric + " | 95% CI | precision | recall | ROC-AUC | avg precision | decision threshold |", "|---|---|---|---|---|---|---|---|---|---|"]
    for suite, r in board.get(exp, {}).items():
        mm = r["metrics"]
        ci = mm.get("ci95")
        ci_s = f"{ci['lo']:.3f}-{ci['hi']:.3f}" if ci else ""
        det.append(f"| {suite} | {_suite_dataset(base, suite)[0]} | {r['n_test']} | {_f(r['objective'])} | {ci_s} | {_f(mm.get('precision'))} | {_f(mm.get('recall'))} | {_f(mm.get('roc_auc'))} | {_f(mm.get('avg_precision'))} | {_f(mm.get('threshold'), 3)} |")
    detail = "\n".join(det)

    labels = "\n".join(f"- **`{l.name}`**: {l.definition}" for l in spec.labels)
    sources = "\n".join(f"- [`{s.dataset}`](https://huggingface.co/datasets/{s.dataset}): role `{s.role}`" + (f", partition `{s.partition}`" if s.partition != "all" else "") for s in spec.data.hf_sources) or "- (none)"
    synth = "- LLM-written examples, double-labelled (generator + blind judge); held-out sets from an independent evaluator prompt\n" if spec.data.synthetic else ""
    lin = _lineage(p, exp)
    lineage = []
    for step in lin:
        if not step["applied"]:
            lineage.append(f"- **{step['exp']}**: initial spec")
        for pr in step["applied"]:
            lineage.append(f"- **{step['exp']}** ← `{pr['id']}` (from {step['parent']}): {pr['why']}")
    lineage_s = "\n".join(lineage)
    lat = board.get(exp, {}).get(prim, {})
    in_domain = [e.name for e in base.evals if e.source and any(s.dataset == e.source.dataset and s.partition == "train" for s in spec.data.hf_sources)]
    caveats = []
    if in_domain:
        caveats.append(f"- Trained on the **train partition** of the source behind {', '.join(in_domain)} (file-level hash split, disjoint from that suite's calib/test files). Results there are in-domain; baselines may not have seen that domain.")
    if nc:
        caveats.append(f"- Training data includes {', '.join(nc)}: **non-commercial**. Check licences before commercial use.")
    sizes = ", ".join(f"{s} ({r['n_test']} test items)" for s, r in board.get(exp, {}).items())
    caveats.append(f"- Evaluated on {sizes}. Small suites have wide confidence intervals.")
    stream = """
### Streaming

```python
s = m.stream()
for chunk in mic_chunks():         # float32 mono, any size
    out = s.step(chunk)            # {"probs": [...per 10ms...], "speaking": bool, "events": [{"type": "start"|"end", "t_s": ...}]}
```
""" if (p.root / "export" / "stream.onnx").exists() else ""
    example = {"audio": 'import soundfile as sf\nwav, sr = sf.read("clip.wav", dtype="float32")   # 16 kHz mono\nprint(m.predict(wav)["segments"])',
               "text": 'print(m.predict("so what I wanted to ask was"))', "image": 'print(m.predict("photo.jpg"))'}[spec.modality]

    body = f"""
# {repo_id.split('/')[-1]}

**{spec.description}**

A tiny, single-purpose *System 1* model built with [BYOJL](https://github.com/bsye/BYOJL): a JEPA-style
encoder trained only for this task, with a calibrated decision policy, shipped as a {m['params'] / 1e6:.2f}M-parameter ONNX
bundle that runs on any CPU with just `numpy` + `onnxruntime`.

| | |
|---|---|
| task | `{spec.modality}` / `{spec.task_type}`{f" (one decision every {spec.frame_hop_ms:g} ms)" if spec.task_type == "frame" else ""} |
| labels | {", ".join(f"`{l}`" for l in spec.label_names)} |
| parameters | {m['params']:,}{f" (shipped as int8 ONNX, {m['quantization']['tried']['+'.join(m['quantization']['ops'])]['mb']:.0f} MB)" if m.get('precision') == 'int8' else ""} |
| backbone | `{spec.backbone}` · embedding dim {m['embedding_dim']} |
| objective | `{base.objective.metric}`, primary suite `{prim}` |
| decision policy | `{m['decision']['kind']}`, threshold {_f(m['decision'].get('threshold'), 3)} (calibrated on `{prim}`'s calib split) |
{f"| CPU, streaming | {_f(lat.get('stream_ms_per_chunk'), 3)} ms per {_f(lat.get('chunk_ms'), 0)} ms chunk, 1 thread (PyTorch; the ONNX bundle is faster) |" if lat.get('stream_ms_per_chunk') else ""}

## Labels

{labels}

## Comparison

Every model below, this one and the baselines, was scored by the same code on the same frozen test splits, with its
decision policy calibrated on the same calib splits. **Bold** = this model.

{comparison}

## Test evaluation (this model)

{detail}

## Usage

```bash
pip install huggingface_hub numpy onnxruntime soundfile
```

```python
from huggingface_hub import snapshot_download
import sys
path = snapshot_download("{repo_id}")
sys.path.insert(0, path)
from byojl_rt import load

m = load(path)
{example}
emb = m.encode(...)                # the learned representation (JEPA-style), for probes / retrieval / other heads
```
{stream}
### Local HTTP server

```bash
python byojl_rt.py serve . --port 8800
# GET /v1/models · POST /v1/{spec.name}/predict · POST /v1/{spec.name}/encode
```

## How it was built

1. **Spec**: labels, definitions, objective and data plan in `spec.yaml` (included in this repo).
2. **Data** ({meta['n_train']:,} training / {meta['n_dev']:,} dev examples):
{synth}{sources}
3. **LeJEPA pretraining** ({spec.train.pretrain_steps} steps): multi-view prediction + SIGReg (isotropic-Gaussian
   regulariser on random projections), no EMA teacher, no stop-gradient.
4. **Supervised training**: tiny head on the backbone, class-weighted, checkpoint chosen on dev (best epoch {meta['best_epoch'] + 1}).
5. **Evaluation**: frozen eval suites; thresholds calibrated on each suite's calib split; baselines scored identically.
6. **Improvement lineage** (each step proposed by `byojl improve` from the previous board):
{lineage_s}

## Limitations

{chr(10).join(caveats)}

## Reproduce

```bash
git clone https://github.com/bsye/BYOJL && cd BYOJL && pip install -e .
# put spec.yaml from this repo at specialists/{spec.name}/spec.yaml, then:
byojl build {spec.name} && byojl board {spec.name} && byojl export {spec.name}
```
"""
    return "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n" + body


def publish(name: str, repo_id: str, license: str = "other", private: bool = True, dry_run: bool = False, log: Callable[[str], None] = print) -> Path:
    p = Project(name)
    exp_dir = p.root / "export"
    if not (exp_dir / "manifest.json").exists():
        raise RuntimeError(f"export first: byojl export {name}")
    m = json.loads((exp_dir / "manifest.json").read_text())
    stage = p.root / "hub" / repo_id.replace("/", "__")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(exp_dir, stage, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    shutil.copy(p.exp_dir(m["experiment"]) / "spec.yaml", stage / "spec.yaml")
    shutil.copy(p.root / "board.json", stage / "board.json")
    lineage = _lineage(p, m["experiment"])
    (stage / "lineage.json").write_text(json.dumps(lineage, indent=2))
    (stage / "README.md").write_text(model_card(name, repo_id, license))
    log(f"  staged {len(list(stage.iterdir()))} files in {stage}")
    if dry_run:
        log("  dry run: nothing uploaded")
        return stage
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo_id, commit_message=f"BYOJL publish {name}/{m['experiment']}")
    log(f"  published https://huggingface.co/{repo_id} ({'private' if private else 'public'})")
    return stage
