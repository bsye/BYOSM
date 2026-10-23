from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..spec import HFSource, TaskSpec
from . import audio_mix, budget, hf_source, panel, segments
from .splits import _group_split3, _hold_out, _split3
from .store import TEST_SHARE, TRAIN_INPUTS, Dataset, dedupe, record, write
from .table import is_table, table_rows

MEDIA_EXT = {"audio": ".wav", "image": ".png", "document": ".png"}
UNLABELED_PER_SOURCE = 2000

Log = Callable[[str], None]


def _save_media(spec: TaskSpec, data_dir: Path, split: str, x: Any) -> str:
    rid = record("", None, "")["id"]
    rel = f"{spec.modality}/{split}/{rid}{MEDIA_EXT[spec.modality]}"
    (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    if spec.modality == "audio":
        import soundfile as sf

        sf.write(str(data_dir / rel), x, spec.data.audio.sample_rate, subtype="PCM_16")
    else:
        x.save(data_dir / rel)
    return rel


def _local(spec: TaskSpec) -> list[tuple[Any, Any, str]]:
    root = Path(spec.data.local_dir or "")
    if is_table(root):
        return table_rows(spec, root)
    out = []
    for lab in spec.label_names:
        for p in sorted((root / lab).glob("*")) if (root / lab).exists() else []:
            if p.suffix == ".txt" and spec.modality == "text":
                out += [(line.strip(), lab, "") for line in p.read_text().splitlines() if line.strip()]
            elif spec.modality != "text":
                out.append((str(p.resolve()), lab, ""))
    return out


def _segment_clips(spec: TaskSpec, src: HFSource, data_dir: Path, log: Log) -> list[dict]:
    import soundfile as sf

    from ..modalities.audio import load_wav

    parts = ("train",) if src.partition in ("train", "all") else (src.partition,)
    files = segments.fetch(src, parts, src.max_files, data_dir.parent.parent / "cache")
    sr, hop = spec.data.audio.sample_rate, int(spec.data.audio.sample_rate * spec.frame_hop_ms / 1000)
    clip = int(sr * spec.data.audio.clip_seconds) // hop * hop
    rows: list[dict] = []
    filt = [b for a, b in src.label_map if a == "keep"] or None
    (data_dir / "audio" / "real").mkdir(parents=True, exist_ok=True)
    for f in files:
        wav = load_wav(f, sr)
        y = segments.frame_labels(f, len(wav) // hop, spec, filt)
        for s0 in range(0, len(wav) - clip + 1, clip):
            if len(rows) >= src.max_rows:
                break
            labels = y[s0 // hop : (s0 + clip) // hop].tolist()
            r = record("", labels, f"hf:{src.dataset}", file=f.name, offset_s=s0 / sr)
            rel = f"audio/real/{r['id']}.wav"
            sf.write(str(data_dir / rel), wav[s0 : s0 + clip], sr, subtype="PCM_16")
            r["input"] = rel
            rows.append(r)
    clip_s = spec.data.audio.clip_seconds
    log(f"  {src.dataset}: {len(rows)} real {clip_s}s training clips from {len(files)} files (partition={parts})")
    return rows


def _split_by_recording(real: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    files = sorted({r["meta"]["file"] for r in real})
    dev_files = set(_split3(files, rng, test=0.0)["dev"]) if len(files) > 1 else set()
    return {
        "train": [r for r in real if r["meta"]["file"] not in dev_files],
        "dev": [r for r in real if r["meta"]["file"] in dev_files],
    }


def _add_local(spec: TaskSpec, splits: dict[str, list[dict]], rng: random.Random, log: Log) -> None:
    local = _local(spec)
    held = [record(x, y, "local") for x, y, s in local if s == "test"]
    rows = [record(x, y, "local") for x, y, s in local if s != "test"]
    parts = _split3(rows, rng, test=0.0 if held else TEST_SHARE)
    parts["test"] += held
    for s, part in parts.items():
        splits[s] += part
    log(f"  local: {len(local)} examples from {spec.data.local_dir} ({len(held)} marked test)")


def _resolve_sources(spec: TaskSpec, data_dir: Path, cfg: dict | None, log: Log) -> tuple[list, list]:
    import yaml

    seg_sources = [s for s in spec.data.hf_sources if s.format == "segment_files"]
    row_spec = spec.model_copy(deep=True)
    row_spec.data.hf_sources = [s for s in spec.data.hf_sources if s.format != "segment_files"]
    sources = hf_source.resolve(row_spec, cfg, log) if (row_spec.data.hf_sources or spec.data.hf_queries) else []
    resolved = data_dir / "sources.resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump([s.model_dump(mode="json") for s in sources + seg_sources], sort_keys=False))
    return sources, seg_sources


def _add_hf(
    spec: TaskSpec, data_dir: Path, sources: list, splits: dict, unlabeled: list, rng: random.Random, log: Log
) -> dict[str, dict[str, list]]:
    pools: dict[str, dict[str, list]] = {
        "positive": {"train": [], "dev": [], "test": []},
        "negative": {"train": [], "dev": [], "test": []},
    }
    for src in sources:
        try:
            rows = hf_source.take(src, spec)
        except Exception as e:
            log(f"  failed to read {src.dataset}: {type(e).__name__}: {str(e)[:160]}")
            continue
        log(f"  {src.dataset}: {len(rows)} rows as {src.role}")
        if src.role == "labeled":
            for s, part in _group_split3([r for r in rows if r[1]], rng).items():
                for x, y in part:
                    inp = x if spec.modality == "text" else _save_media(spec, data_dir, s, x)
                    splits[s].append(record(inp, y, f"hf:{src.dataset}"))
        elif src.role in pools and spec.modality == "audio":
            for s, part in _split3([x for x, _ in rows], rng).items():
                pools[src.role][s] += part
        else:
            cap = max(UNLABELED_PER_SOURCE, spec.data.teacher_rows if spec.data.teachers else 0)
            for x, _ in rows[:cap]:
                inp = x if spec.modality == "text" else _save_media(spec, data_dir, "unlabeled", x)
                unlabeled.append(record(inp, None, f"hf:{src.dataset}"))
    return pools


def _collect(
    spec: TaskSpec, data_dir: Path, cfg: dict | None, seconds: float, log: Log
) -> tuple[dict[str, list[dict]], list[dict], dict | None]:
    rng = random.Random(spec.name)
    fit = None
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    unlabeled: list[dict] = []
    if spec.data.local_dir:
        _add_local(spec, splits, rng, log)
    sources, seg_sources = _resolve_sources(spec, data_dir, cfg, log)
    pools = _add_hf(spec, data_dir, sources, splits, unlabeled, rng, log)
    for src in seg_sources:
        for s, rows in _split_by_recording(_segment_clips(spec, src, data_dir, log), rng).items():
            splits[s] += rows
    if spec.modality == "audio" and pools["positive"]["train"]:
        sizes = {"train": spec.data.train_size, "dev": spec.data.dev_size, "test": spec.data.test_size}
        for s, rows in audio_mix.build(spec, data_dir, pools["positive"], pools["negative"], sizes, log).items():
            splits[s] += rows
    if spec.data.synthetic and spec.modality == "text":
        fit = budget.add_synthetic(spec, splits, cfg, seconds, log, data_dir)
    _hold_out(splits, rng, log)
    if spec.modality == "text":
        splits["test"] = dedupe(splits["test"], against=splits["train"] + splits["dev"])
        splits["dev"] = dedupe(splits["dev"], against=splits["train"])
    return splits, unlabeled, fit


def build(
    spec: TaskSpec,
    data_dir: Path,
    cfg: dict | None = None,
    log: Callable[[str], None] = print,
    reviewed: dict[str, str] | None = None,
    seconds: float | None = None,
) -> dict[str, Any]:
    ds = Dataset(data_dir)
    allowed = spec.data.max_minutes * 60 if seconds is None else seconds
    splits, unlabeled, fit = _collect(spec, data_dir, cfg, allowed, log)
    if not splits["train"]:
        raise RuntimeError(
            "data stage produced no training examples - check the data plan in spec.yaml (sources, synthetic, local_dir), "
            "or run `byosm request-data <name>` for a template to fill"
        )
    if spec.data.teachers:
        labelled = panel.label_pool(spec, splits, unlabeled, data_dir, log, reviewed)
        held = splits["dev"] + splits["test"]
        splits["train"] += dedupe(labelled, against=held) if spec.modality == "text" else labelled
    for s, rows in splits.items():
        ds.save(s, rows)
    unlabeled += [record(r["input"], None, TRAIN_INPUTS) for r in splits["train"]]
    write(data_dir / "unlabeled.jsonl", unlabeled)
    if fit:
        budget.record_fit(data_dir, fit, splits)
    return ds.summary()
