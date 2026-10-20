"""One interface for everything that gets scored: our experiments and external baselines.

A Predictor returns probabilities over the spec's labels:
  classify -> probs_batch(raws) -> [N, C]
  frame    -> frame_probs(raw)  -> [T, C] at the spec's frame hop

Add a baseline with @register_baseline("name"). It is then usable from spec.yaml (`baselines: [name]`) and from the CLI.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .spec import TaskSpec


class Predictor:
    id: str = ""
    kind: str = "baseline"
    params: int | None = None
    modalities: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ("classify", "frame")

    def __init__(self, spec: TaskSpec):
        self.spec = spec

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        raise NotImplementedError

    def frame_probs(self, raw: Any) -> np.ndarray:
        raise NotImplementedError

    def stream_chunk_ms(self) -> float | None:
        """CPU time to process one streaming chunk (frame tasks), if the predictor streams."""
        return None

    chunk_ms: float | None = None

    def latency_ms(self, x: Any) -> float:
        for _ in range(3):
            self._one(x)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            self._one(x)
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))

    def _one(self, x: Any) -> Any:
        return self.frame_probs(x) if self.spec.task_type == "frame" else self.probs_batch([x])


class ExperimentPredictor(Predictor):
    kind = "experiment"

    def __init__(self, spec: TaskSpec, run_dir: Path, name: str, device: str = "cpu"):
        super().__init__(spec)
        from .runtime import Specialist

        self.sp = Specialist(run_dir, device)
        self.device = device
        self.id = name
        self.params = int(self.sp.meta["params"])
        self.bb = self.sp.model.backbone
        self.head = next(iter(self.sp.model.heads.values()))
        self.chunk_ms = 30.0 if hasattr(self.bb, "stream") else None

    @torch.no_grad()
    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = []
        for i in range(0, len(raws), 64):
            inputs, _ = self.sp.mod.collate(raws[i : i + 64])
            logits, _ = self.sp.model({k: v.to(self.device) for k, v in inputs.items()})
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def frame_probs(self, raw: Any) -> np.ndarray:
        if hasattr(self.bb, "stream"):  # continuous streaming, like deployment
            hop = self.bb.hop
            n = len(raw) // hop * hop
            state, out = None, []
            for i in range(0, n, 160_000):
                f, state = self.bb.stream(torch.from_numpy(np.ascontiguousarray(raw[i : min(n, i + 160_000)]))[None], state)
                out.append(torch.softmax(self.head.net(f), -1)[0].numpy())
            return np.concatenate(out) if out else np.zeros((0, len(self.spec.labels)), np.float32)
        # Non-streaming backbones: independent windows of the training clip length.
        sr = self.spec.data.audio.sample_rate
        w = int(sr * self.spec.data.audio.clip_seconds)
        hop = int(sr * self.spec.frame_hop_ms / 1000)
        out = []
        for i in range(0, len(raw), w):
            seg = raw[i : i + w]
            if len(seg) < hop:
                break
            inputs, _ = self.sp.mod.collate([seg])
            logits, mask = self.sp.model(inputs)
            p = torch.softmax(logits.float(), -1)[0][mask[0]].numpy()
            idx = np.minimum((np.arange(len(seg) // hop) * len(p)) // max(1, len(seg) // hop), len(p) - 1)
            out.append(p[idx])
        return np.concatenate(out)

    @torch.no_grad()
    def stream_chunk_ms(self) -> float | None:
        if not hasattr(self.bb, "stream"):
            return None
        x = torch.from_numpy(np.random.default_rng(0).normal(0, 0.05, (1, 480)).astype(np.float32))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # single core, like Silero in LiveKit
        try:
            _, state = self.bb.stream(x)
            t = time.perf_counter()
            for _ in range(1000):
                f, state = self.bb.stream(x, state)
                self.head.net(f)
            return time.perf_counter() - t
        finally:
            torch.set_num_threads(threads)


BASELINES: dict[str, Callable[[TaskSpec], Predictor]] = {}


def register_baseline(name: str) -> Callable[[type[Predictor]], type[Predictor]]:
    def deco(cls: type[Predictor]) -> type[Predictor]:
        cls.id = name
        BASELINES[name] = cls
        return cls

    return deco


def get_baseline(name: str, spec: TaskSpec) -> Predictor:
    if name.startswith("hf:"):
        return HFClassifier(spec, name[3:])
    if name.startswith("llm:"):  # another LLM role from byosm.yaml, e.g. llm:gpt -> llm.gpt
        return ZeroShotLLM(spec, role=name[4:])
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    cls = BASELINES[name]
    if cls.modalities and spec.modality not in cls.modalities or spec.task_type not in cls.task_types:
        raise ValueError(f"baseline {name!r} supports {cls.modalities} / {cls.task_types}, not {spec.modality} / {spec.task_type}")
    return cls(spec)


@register_baseline("silero")
class SileroVAD(Predictor):
    """Silero VAD, the exact ONNX bundled with livekit-plugins-silero (LiveKit's VAD)."""

    modalities, task_types = ("audio",), ("frame",)
    window = 512
    chunk_ms = 32.0

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        try:
            from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session
        except ImportError as e:
            raise ImportError("pip install livekit-plugins-silero onnxruntime") from e
        self.m = OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=16000)
        self.params = 545_601  # both 8k and 16k branches of the bundled ONNX

    def frame_probs(self, raw: Any) -> np.ndarray:
        self.m.reset()
        n = len(raw) // self.window
        wp = np.array([self.m(np.ascontiguousarray(raw[i * self.window : (i + 1) * self.window])[None]) for i in range(n)], np.float32)
        hop = int(16000 * self.spec.frame_hop_ms / 1000)
        # frame -> window containing the frame start (up to 32ms look-ahead: favours Silero)
        idx = np.minimum(np.arange(len(raw) // hop) * hop // self.window, max(0, len(wp) - 1))
        p = wp[idx] if len(wp) else np.zeros(len(idx), np.float32)
        out = np.zeros((len(p), len(self.spec.labels)), np.float32)
        out[:, self.spec.pos_index] = p
        out[:, [i for i in range(len(self.spec.labels)) if i != self.spec.pos_index][0]] = 1 - p
        return out

    def stream_chunk_ms(self) -> float:
        self.m.reset()
        x = np.random.default_rng(0).normal(0, 0.05, (1, self.window)).astype(np.float32)
        t = time.perf_counter()
        for _ in range(1000):
            self.m(x)
        return time.perf_counter() - t


@register_baseline("energy")
class EnergyVAD(Predictor):
    """Floor baseline: frame log-energy squashed to [0,1]. If a model can't beat this, something is wrong."""

    modalities, task_types = ("audio",), ("frame",)
    params = 0

    def frame_probs(self, raw: Any) -> np.ndarray:
        hop = int(16000 * self.spec.frame_hop_ms / 1000)
        n = len(raw) // hop
        e = 10 * np.log10(np.mean(np.asarray(raw[: n * hop]).reshape(n, hop) ** 2, 1) + 1e-10)
        p = 1 / (1 + np.exp(-(e + 45) / 4))
        out = np.zeros((n, len(self.spec.labels)), np.float32)
        out[:, self.spec.pos_index] = p
        out[:, [i for i in range(len(self.spec.labels)) if i != self.spec.pos_index][0]] = 1 - p
        return out


@register_baseline("llm")
class ZeroShotLLM(Predictor):
    """The LLM itself, zero-shot, given the label definitions (role `baseline` in byosm.yaml): the System-2 bar.
    Text: the input. Documents: each page as a readable image (~1000px) + its text layer / OCR. Images: the image."""

    modalities, task_types = ("text", "document", "image"), ("classify",)
    wants_rows = True  # needs the original files (full resolution), not the model's preprocessed inputs
    batch = {"text": 40, "document": 4, "image": 6}

    def __init__(self, spec: TaskSpec, role: str = "baseline"):
        super().__init__(spec)
        from .llm import get_llm

        self.llm = get_llm(role)
        self.id = "llm" if role == "baseline" else f"llm:{role}"
        self.name = self.llm.name

    def _item(self, row: dict, mod: Any) -> tuple[str, list[bytes]]:
        import io

        if self.spec.modality == "text":
            return str(row["input"]), []
        path = Path(row["input"])
        if not path.is_absolute() and getattr(mod, "data_dir", None):
            path = Path(mod.data_dir) / path
        if self.spec.modality == "document":
            from .modalities.document import read_pages

            pages = read_pages(Path(path), self.spec.document.max_pages, max(self.spec.document.dpi, 100), self.spec.document.ocr)[0]
        else:
            from PIL import Image

            pages = [("", Image.open(path).convert("RGB"))]
        imgs = []
        for _, im in pages:
            im = im.copy()
            im.thumbnail((1000, 1000))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            imgs.append(buf.getvalue())
        text = "\n".join(f"[page {i + 1} text] {t[:1500]}" for i, (t, _) in enumerate(pages) if t.strip())
        return text, imgs

    def probs_rows(self, rows: list[dict], mod: Any) -> np.ndarray:
        from .data.synth_text import _context
        from .llm import pmap

        names = self.spec.label_names
        schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": names}}}}
        system = ("You are an expert annotator. Classify each item strictly according to the label definitions. "
                  "Return exactly one label per item, in order.")
        bs = self.batch[self.spec.modality]
        chunks = [rows[i : i + bs] for i in range(0, len(rows), bs)]

        def run(chunk: list[dict]) -> list[str]:
            items, images = [], []
            for j, r in enumerate(chunk):
                text, imgs = self._item(r, mod)
                first = len(images) + 1
                images += imgs
                ref = f" (images {first}-{len(images)})" if len(imgs) > 1 else (f" (image {first})" if imgs else "")
                items.append(f"Item {j + 1}{ref}:" + (f"\n{text}" if text else ""))
            prompt = f"{_context(self.spec)}\n\n" + "\n\n".join(items) + f"\n\nReturn {len(chunk)} labels, one per item, in order."
            kw = {"images": images} if images else {}  # text-only providers needn't accept `images`
            labs = self.llm.json(system, prompt, schema, **kw).get("labels", [])
            return (labs + [names[0]] * len(chunk))[: len(chunk)]

        labs = [l for part in pmap(run, chunks, 8) for l in part]
        out = np.zeros((len(rows), len(names)), np.float32)
        for i, l in enumerate(labs):
            out[i, names.index(l) if l in names else 0] = 1.0
        return out

    def probs_batch(self, raws: list[Any]) -> np.ndarray:  # text inputs only (latency checks)
        return self.probs_rows([{"input": r} for r in raws], None)

    def latency_ms(self, x: Any) -> float:
        return float("nan")  # network-bound; not comparable to local models


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


class HFClassifier(Predictor):
    """Any Hugging Face classifier as a baseline: `hf:<repo>`. Image/document models see the (first) page image,
    text models the text. Its labels are matched to ours by name; probability on unmatched labels is dropped."""

    def __init__(self, spec: TaskSpec, repo: str):
        super().__init__(spec)
        from transformers import AutoConfig

        self.id = f"hf:{repo}"
        cfg = AutoConfig.from_pretrained(repo)
        self.visual = spec.modality in ("image", "document")
        if self.visual:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.proc = AutoImageProcessor.from_pretrained(repo)
            self.model = AutoModelForImageClassification.from_pretrained(repo).eval()
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.proc = AutoTokenizer.from_pretrained(repo)
            self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        ours = {_norm(n): i for i, n in enumerate(spec.label_names)}
        self.map = {int(k): ours[_norm(v)] for k, v in cfg.id2label.items() if _norm(v) in ours}
        missing = set(range(len(spec.labels))) - set(self.map.values())
        if not self.map:
            raise ValueError(f"{repo}: none of its labels {list(cfg.id2label.values())[:8]} match ours {spec.label_names}")
        self.missing = [spec.label_names[i] for i in sorted(missing)]
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        out = []
        for i in range(0, len(raws), 32):
            chunk = raws[i : i + 32]
            if self.visual:
                imgs = [r[0].image if isinstance(r, list) else r for r in chunk]  # documents: first page
                inputs = self.proc(images=imgs, return_tensors="pt")
            else:
                inputs = self.proc(chunk, padding=True, truncation=True, return_tensors="pt")
            p = torch.softmax(self.model(**inputs).logits.float(), -1).numpy()
            q = np.zeros((len(chunk), len(self.spec.labels)), np.float32)
            for src, dst in self.map.items():
                q[:, dst] += p[:, src]
            out.append(q / np.maximum(q.sum(1, keepdims=True), 1e-9))
        return np.concatenate(out)
