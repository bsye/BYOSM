"""Data stage: turn the spec's data plan into data/{train,dev,test,unlabeled}.jsonl."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from ..spec import TaskSpec
from . import audio_mix, hf_source, segments, synth_text
from .store import Dataset, dedupe, record, write

MEDIA_EXT = {"audio": ".wav", "image": ".png", "document": ".png"}


def _split3(items: list, rng: random.Random, dev: float = 0.1, test: float = 0.15) -> dict[str, list]:
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


def _local(spec: TaskSpec) -> list[tuple[Any, str]]:
    """<local_dir>/<label>/* : .txt (one example per line) or media files."""
    out = []
    root = Path(spec.data.local_dir or "")
    for lab in spec.label_names:
        for p in sorted((root / lab).glob("*")) if (root / lab).exists() else []:
            if p.suffix == ".txt" and spec.modality == "text":
                out += [(l.strip(), lab) for l in p.read_text().splitlines() if l.strip()]
            elif spec.modality != "text":
                out.append((str(p.resolve()), lab))
    return out


def _segment_clips(spec: TaskSpec, src, data_dir: Path, log: Callable[[str], None]) -> list[dict]:  # noqa: ANN001
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
            r = record("", y[s0 // hop : (s0 + clip) // hop].tolist(), f"hf:{src.dataset}", file=f.name, offset_s=s0 / sr)
            rel = f"audio/real/{r['id']}.wav"
            sf.write(str(data_dir / rel), wav[s0 : s0 + clip], sr, subtype="PCM_16")
            r["input"] = rel
            rows.append(r)
    log(f"  {src.dataset}: {len(rows)} real {spec.data.audio.clip_seconds}s training clips from {len(files)} files (partition={parts})")
    return rows


def build(spec: TaskSpec, data_dir: Path, cfg: dict | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    ds = Dataset(data_dir)
    rng = random.Random(spec.name)
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    unlabeled: list[dict] = []

    # 1. Real data: local folders and Hugging Face.
    if spec.data.local_dir:
        rows = [record(x, y, "local") for x, y in _local(spec)]
        for s, part in _split3(rows, rng).items():
            splits[s] += part
        log(f"  local: {len(rows)} examples from {spec.data.local_dir}")

    seg_sources = [s for s in spec.data.hf_sources if s.format == "segment_files"]
    row_spec = spec.model_copy(deep=True)
    row_spec.data.hf_sources = [s for s in spec.data.hf_sources if s.format == "rows"]
    sources = hf_source.resolve(row_spec, cfg, log) if (row_spec.data.hf_sources or spec.data.hf_queries) else []
    import yaml

    (data_dir / "sources.resolved.yaml").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "sources.resolved.yaml").write_text(yaml.safe_dump([s.model_dump(mode="json") for s in sources + seg_sources], sort_keys=False))
    pools: dict[str, dict[str, list]] = {"positive": {"train": [], "dev": [], "test": []}, "negative": {"train": [], "dev": [], "test": []}}
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
        else:  # unlabeled
            for x, _ in rows[:2000]:
                unlabeled.append(record(x if spec.modality == "text" else _save_media(spec, data_dir, "unlabeled", x), None, f"hf:{src.dataset}"))

    # 1b. Real segment-labelled audio (e.g. AVA-Speech train partition), chopped into training clips.
    for src in seg_sources:
        real = _segment_clips(spec, src, data_dir, log)
        splits["dev"] += real[::10]  # some real audio in dev so model selection sees it too
        splits["train"] += [r for i, r in enumerate(real) if i % 10]

    # 2. Procedural audio from positive/negative pools (exact labels by construction).
    if spec.modality == "audio" and pools["positive"]["train"]:
        sizes = {"train": spec.data.train_size, "dev": spec.data.dev_size, "test": spec.data.test_size}
        for s, rows in audio_mix.build(spec, data_dir, pools["positive"], pools["negative"], sizes, log).items():
            splits[s] += rows

    # 3. LLM-written text: train from the generator, dev/test from the independent evaluator.
    if spec.data.synthetic and spec.modality == "text":
        need = max(0, spec.data.train_size - len(splits["train"]))
        if need:
            splits["train"] += synth_text.generate(spec, need, "train", cfg, against=splits["train"], log=log)
        for s, size in (("dev", spec.data.dev_size), ("test", spec.data.test_size)):
            need = max(0, size - len(splits[s]))
            if need:
                # Dedupe against everything already written so held-out sets can't leak from train.
                prior = splits["train"] + splits["dev"] + splits["test"]
                splits[s] += synth_text.generate(spec, need, s, cfg, against=prior, log=log)

    if spec.modality == "text":
        splits["test"] = dedupe(splits["test"], against=splits["train"] + splits["dev"])
        splits["dev"] = dedupe(splits["dev"], against=splits["train"])

    if not splits["train"]:  # fail before writing anything, so an empty dataset is never cached
        raise RuntimeError("data stage produced no training examples - check the data plan in spec.yaml (sources, synthetic, local_dir)")
    for s, rows in splits.items():
        ds.save(s, rows)
    unlabeled += [record(r["input"], None, "train-inputs") for r in splits["train"]]
    write(data_dir / "unlabeled.jsonl", unlabeled)
    return ds.summary()
