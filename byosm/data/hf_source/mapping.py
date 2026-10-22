from __future__ import annotations

from collections.abc import Callable

from ...llm import get_llm, parallel_calls, pmap, role_model
from ...spec import HFSource, TaskSpec
from .cards import describe, listing
from .hub import _features, closest, search

MAP_SYSTEM = """You map a Hugging Face dataset onto a classifier's label space. Pick the config/split, the input column \
(the raw text/audio/image), and for role=labeled the label column plus an explicit mapping from source label values to \
our labels (omit source values that do not fit any of our labels cleanly - precision beats coverage). For role \
positive/negative/unlabeled only the input column matters (label_map may filter rows: map wanted source values to "keep"). \
If no column holds labels for role=labeled, investigate whether our labels follow from the content instead: give label_rules as [column, text that column contains, our label], checked in order, and label_default for rows that match none (leave it empty to drop them). Columns may be paths into nested values, like conversations.1.value. The input column must not contain the text the rules look for, or the model learns the answer from the input. Set usable=false only if neither a label column nor reliable rules fit the role."""

PICK_ROLE, MAP_ROLE = "pick", "map"
RULE_PARTS = 3
PICK_SYSTEM = """You choose Hugging Face datasets to build a training set. Given search results, select the datasets \
(by exact id) that best provide data in each role. Prefer widely-used, permissively licensed, parquet-backed datasets."""


def _pick_schema() -> dict:
    role = {"type": "string", "enum": ["labeled", "positive", "negative", "unlabeled"]}
    pick = {"type": "object", "properties": {"dataset": {"type": "string"}, "role": role, "why": {"type": "string"}}}
    return {"type": "object", "properties": {"picks": {"type": "array", "items": pick}}}


def _map_schema() -> dict:
    pair = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean"},
            "config": {"type": "string"},
            "split": {"type": "string"},
            "input_column": {"type": "string"},
            "label_column": {"type": "string"},
            "label_map": {"type": "array", "items": pair},
            "label_rules": {"type": "array", "items": pair},
            "label_default": {"type": "string"},
        },
    }


def pick_sources(spec: TaskSpec, hits: list[dict], cfg: dict | None) -> list[tuple[HFSource, str]]:
    if not hits:
        return []
    prompt = (
        f"Task: {spec.summary or spec.description}\nModality: {spec.modality}\nLabels: {spec.label_names}\n\n"
        f"Search results:\n{listing(describe(hits))}"
    )
    out = get_llm(PICK_ROLE, cfg).json(PICK_SYSTEM, prompt, _pick_schema())
    ids = {h["id"] for h in hits}
    picks = [p for p in out.get("picks", []) if p.get("dataset") in ids and p.get("role")]
    return [(HFSource(dataset=p["dataset"], role=p["role"]), str(p.get("why") or "")) for p in picks]


def _pick_sources(spec: TaskSpec, cfg: dict | None) -> list[HFSource]:
    return [src for src, _ in pick_sources(spec, search(spec.data.hf_queries), cfg)]


def _map_prompt(
    spec: TaskSpec, src: HFSource, feats: dict, configs: list[str], preview: dict, splits: list[str]
) -> str:
    return (
        f"Our task: {spec.summary or spec.description}\nOur labels: {[(lb.name, lb.definition) for lb in spec.labels]}\n"
        f"Role of this dataset: {src.role}\nDataset: {src.dataset}\nConfigs: {configs[:30]}\nSplits: {splits}\n"
        f"Features: {feats}\nFirst row: {preview}"
    )


def _root(path: str | None) -> str:
    return (path or "").split(".")[0]


def _rules(out: dict, feats: dict, labels: list[str]) -> list[list[str]]:
    return [
        r
        for r in out.get("label_rules", [])
        if len(r) == RULE_PARTS and _root(r[0]) in feats and r[1] and r[2] in labels
    ]


def _apply_mapping(
    src: HFSource, out: dict, feats: dict, configs: list[str], splits: list[str], labels: list[str]
) -> HFSource:
    config = (out.get("config") or src.config) if out.get("config") in configs else src.config
    label_column = (out.get("label_column") or None) if out.get("label_column") in feats else None
    return src.model_copy(
        update={
            "config": config,
            "split": out.get("split") if out.get("split") in splits else splits[0],
            "input_column": out["input_column"],
            "label_column": label_column,
            "label_map": [m for m in out.get("label_map", []) if len(m) == 2],
            "label_rules": _rules(out, feats, labels),
            "label_default": out.get("label_default") if out.get("label_default") in labels else None,
        }
    )


def _map_source(src: HFSource, spec: TaskSpec, cfg: dict | None, log: Callable[[str], None]) -> HFSource | None:
    real = closest(src.dataset)
    if real is None:
        log(f"  skip {src.dataset}: not on the Hub, and no dataset there has that name")
        return None
    if real != src.dataset:
        log(f"  {src.dataset} is not on the Hub; using {real}, the closest name")
        src = src.model_copy(update={"dataset": real})
    try:
        feats, configs, preview, splits = _features(src)
    except Exception as e:
        log(f"  skip {src.dataset}: {type(e).__name__}: {str(e)[:120]}")
        return None
    prompt = _map_prompt(spec, src, feats, configs, preview, splits)
    log(f"  reading the columns of {src.dataset} with {role_model(MAP_ROLE, cfg)}")
    out = get_llm(MAP_ROLE, cfg).json(MAP_SYSTEM, prompt, _map_schema())
    if not out.get("usable", True) or _root(out.get("input_column")) not in feats:
        log(f"  skip {src.dataset}: not usable for role {src.role}")
        return None
    src = _apply_mapping(src, out, feats, configs, splits, spec.label_names)
    rules = f" rules={len(src.label_rules)} default={src.label_default}" if src.label_rules else ""
    log(
        f"  mapped {src.dataset} [{src.config or '-'}:{src.split}] role={src.role} input={src.input_column} "
        f"label={src.label_column} map={len(src.label_map)}{rules}"
    )
    return src


def resolve(spec: TaskSpec, cfg: dict | None = None, log: Callable[[str], None] = print) -> list[HFSource]:
    sources = list(spec.data.hf_sources)
    if not sources and spec.data.hf_queries:
        sources = _pick_sources(spec, cfg)
    found = pmap(lambda s: s if s.input_column else _map_source(s, spec, cfg, log), sources, parallel_calls(cfg))
    return [s for s in found if s is not None]
