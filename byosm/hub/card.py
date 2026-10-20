from __future__ import annotations

import yaml

from ..format import fmt
from . import sections
from .facts import CardFacts, load_facts
from .tables import comparison_table, detail_table, front_matter

INTRO = """
# {repo_name}

**{description}**

A single-purpose *System 1* model built with [BYOSM](https://github.com/bsye/BYOSM): {built}, with a calibrated
decision policy, shipped as a {mparams:.2f}M-parameter ONNX bundle that runs on any CPU with `onnxruntime`.

"""
RESULTS = """
## Labels

{labels}

## Comparison

Every model below, this one and the baselines, was scored by the same code on the same frozen test splits, with its
decision policy calibrated on the same calib splits. **Bold** = this model.

{comparison}

## Test evaluation (this model)

{detail}
"""
USAGE = """
## Usage

```bash
pip install {pip}
```

```python
from huggingface_hub import snapshot_download
import sys
path = snapshot_download("{repo_id}")
sys.path.insert(0, path)
from byosm_rt import load

m = load(path)
{example}
emb = m.encode(...)                # the learned representation (JEPA-style), for probes / retrieval / other heads
```
{stream}
### Local HTTP server

```bash
python byosm_rt.py serve . --port 8800
# GET /v1/models · POST /v1/{name}/predict · POST /v1/{name}/encode
```
"""
BUILT = """
## How it was built

1. **Spec**: labels, definitions, objective and data plan in `spec.yaml` (included in this repo).
2. **Data** ({n_train:,} training / {n_dev:,} dev examples):
{synth}{sources}
{steps} (each step proposed by `byosm improve` from the previous board):
{lineage}

{notes}
## Limitations

{caveats}

## Reproduce

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM && pip install -e .
# put spec.yaml from this repo at specialists/{name}/spec.yaml, then:
byosm build {name} && byosm board {name} && byosm export {name}
```
"""


def _backbone(f: CardFacts) -> str:
    spec = f.spec
    if spec.modality == "document":
        doc = spec.document
        return f"text `{doc.text_backbone}` + visual `{doc.image_backbone}`, route `{doc.route}`"
    return f"`{spec.backbone}`"


def _params_cell(m: dict) -> str:
    if m.get("precision") != "int8":
        return f"{m['params']:,}"
    q = m["quantization"]
    return f"{m['params']:,} (shipped as int8 ONNX, {q['tried']['+'.join(q['ops'])]['mb']:.0f} MB)"


def _latency_row(f: CardFacts) -> str:
    lat = f.results.get(f.prim, {})
    if not lat.get("stream_ms_per_chunk"):
        return ""
    ms, chunk = fmt(lat.get("stream_ms_per_chunk"), 3), fmt(lat.get("chunk_ms"), 0)
    return f"| CPU, streaming | {ms} ms per {chunk} ms chunk, 1 thread (PyTorch; the ONNX bundle is faster) |"


def summary_table(f: CardFacts) -> str:
    spec, m = f.spec, f.manifest
    hop = f" (one decision every {spec.frame_hop_ms:g} ms)" if spec.task_type == "frame" else ""
    dec = m["decision"]
    rows = [
        "| | |",
        "|---|---|",
        f"| task | `{spec.modality}` / `{spec.task_type}`{hop} |",
        f"| labels | {', '.join(f'`{lab}`' for lab in spec.label_names)} |",
        f"| parameters | {_params_cell(m)} |",
        f"| backbone | {_backbone(f)} · embedding dim {m['embedding_dim']} |",
        f"| objective | `{f.base.objective.metric}`, primary suite `{f.prim}` |",
        f"| decision policy | `{dec['kind']}`, threshold {fmt(dec.get('threshold'), 3)} (calibrated on `{f.prim}`'s calib split) |",
        _latency_row(f),
    ]
    return "\n".join(rows) + "\n"


def _body(f: CardFacts) -> str:
    spec = f.spec
    intro = INTRO.format(
        repo_name=f.repo_name, description=spec.description, built=sections.built(f), mparams=f.manifest["params"] / 1e6
    )
    results = RESULTS.format(labels=sections.labels(f), comparison=comparison_table(f), detail=detail_table(f))
    usage = USAGE.format(
        pip=sections.pip_packages(f),
        repo_id=f.repo_id,
        example=sections.example(f),
        stream=sections.stream(f),
        name=spec.name,
    )
    built = BUILT.format(
        n_train=f.meta["n_train"],
        n_dev=f.meta["n_dev"],
        synth=sections.synth(f),
        sources=sections.sources(f),
        steps=sections.build_steps(f),
        lineage=sections.lineage_list(f),
        notes=sections.notes(f),
        caveats=sections.caveats(f),
        name=spec.name,
    )
    return intro + summary_table(f) + results + usage + built


def model_card(name: str, repo_id: str, license: str) -> str:
    f = load_facts(name, repo_id, license)
    return "---\n" + yaml.safe_dump(front_matter(f), sort_keys=False, allow_unicode=True) + "---\n" + _body(f)
