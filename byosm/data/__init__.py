from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..spec import HFSource, TaskSpec
from . import audio_mix, hf_source, segments, synth_text
from .store import DEV_SHARE, TEST_SHARE, Dataset, dedupe, record, write
from .table import is_table, table_rows

MEDIA_EXT = {"audio": ".wav", "image": ".png", "document": ".png"}

Log = Callable[[str], None]


def _split3(items: list, rng: random.Random, dev: float = DEV_SHARE, test: float = TEST_SHARE) -> dict[str, list]:
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    a, b = int(n * (1 - dev - test)), int(n * (1 - test))
    return {"train": items[:a], "dev": items[a:b], "test": items[b:]}


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
            for s, part in _split3([r for r in rows if r[1]], rng).items():
                for x, y in part:
                    inp = x if spec.modality == "text" else _save_media(spec, data_dir, s, x)
                    splits[s].append(record(inp, y, f"hf:{src.dataset}"))
        elif src.role in pools:
            for s, part in _split3([x for x, _ in rows], rng).items():
                pools[src.role][s] += part
        else:
            for x, _ in rows[:2000]:
                inp = x if spec.modality == "text" else _save_media(spec, data_dir, "unlabeled", x)
                unlabeled.append(record(inp, None, f"hf:{src.dataset}"))
    return pools


def _add_synthetic(spec: TaskSpec, splits: dict[str, list[dict]], cfg: dict | None, log: Log) -> None:
    need = max(0, spec.data.train_size - len(splits["train"]))
    if need:
        splits["train"] += synth_text.generate(spec, need, "train", cfg, against=splits["train"], log=log)
    for s, size in (("dev", spec.data.dev_size), ("test", spec.data.test_size)):
        need = max(0, size - len(splits[s]))
        if need:
            prior = splits["train"] + splits["dev"] + splits["test"]
            splits[s] += synth_text.generate(spec, need, s, cfg, against=prior, log=log)


def _collect(spec: TaskSpec, data_dir: Path, cfg: dict | None, log: Log) -> tuple[dict[str, list[dict]], list[dict]]:
    rng = random.Random(spec.name)
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    unlabeled: list[dict] = []
    if spec.data.local_dir:
        _add_local(spec, splits, rng, log)
    sources, seg_sources = _resolve_sources(spec, data_dir, cfg, log)
    pools = _add_hf(spec, data_dir, sources, splits, unlabeled, rng, log)
    for src in seg_sources:
        real = _segment_clips(spec, src, data_dir, log)
        splits["dev"] += real[::10]
        splits["train"] += [r for i, r in enumerate(real) if i % 10]
    if spec.modality == "audio" and pools["positive"]["train"]:
        sizes = {"train": spec.data.train_size, "dev": spec.data.dev_size, "test": spec.data.test_size}
        for s, rows in audio_mix.build(spec, data_dir, pools["positive"], pools["negative"], sizes, log).items():
            splits[s] += rows
    if spec.data.synthetic and spec.modality == "text":
        _add_synthetic(spec, splits, cfg, log)
    if spec.modality == "text":
        splits["test"] = dedupe(splits["test"], against=splits["train"] + splits["dev"])
        splits["dev"] = dedupe(splits["dev"], against=splits["train"])
    return splits, unlabeled


def build(
    spec: TaskSpec, data_dir: Path, cfg: dict | None = None, log: Callable[[str], None] = print
) -> dict[str, Any]:
    ds = Dataset(data_dir)
    splits, unlabeled = _collect(spec, data_dir, cfg, log)
    if not splits["train"]:
        raise RuntimeError(
            "data stage produced no training examples - check the data plan in spec.yaml (sources, synthetic, local_dir), "
            "or run `byosm request-data <name>` for a template to fill"
        )
    for s, rows in splits.items():
        ds.save(s, rows)
    unlabeled += [record(r["input"], None, "train-inputs") for r in splits["train"]]
    write(data_dir / "unlabeled.jsonl", unlabeled)
    return ds.summary()
