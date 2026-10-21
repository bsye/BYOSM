from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np


def hysteresis(
    p: np.ndarray,
    hop_s: float,
    act: float = 0.5,
    deact_offset: float = 0.15,
    min_on_ms: float = 50.0,
    min_off_ms: float = 550.0,
    alpha_per_32ms: float = 0.35,
) -> np.ndarray:
    st = HysteresisState(hop_s, act, deact_offset, min_on_ms, min_off_ms, alpha_per_32ms)
    for x in p:
        st.push(float(x))
    return st.finish(len(p))


class HysteresisState:
    def __init__(
        self,
        hop_s: float,
        act: float = 0.5,
        deact_offset: float = 0.15,
        min_on_ms: float = 50.0,
        min_off_ms: float = 550.0,
        alpha_per_32ms: float = 0.35,
    ):
        self.hop, self.act, self.deact = hop_s, act, max(act - deact_offset, 0.01)
        self.min_on, self.min_off = min_on_ms / 1000, min_off_ms / 1000
        self.a = alpha_per_32ms ** (hop_s / 0.032)
        self.filt: float | None = None
        self.speaking, self.on, self.off, self.i, self.start = False, 0.0, 0.0, 0, 0
        self.regions: list[tuple[int, int]] = []

    def push(self, x: float) -> str | None:
        self.filt = x if self.filt is None else self.a * self.filt + (1 - self.a) * x
        ev = None
        if self.filt >= self.act or (self.speaking and self.filt > self.deact):
            self.on += self.hop
            self.off = 0.0
            if not self.speaking and self.on >= self.min_on - 1e-9:
                self.speaking, self.start, ev = True, self.i - int(round(self.on / self.hop)) + 1, "start"
        else:
            self.off += self.hop
            self.on = 0.0
            if self.speaking and self.off >= self.min_off - 1e-9:
                self.speaking, ev = False, "end"
                self.regions.append((self.start, self.i - int(round(self.off / self.hop)) + 1))
        self.i += 1
        return ev

    def finish(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        for a, b in self.regions + ([(self.start, n)] if self.speaking else []):
            out[max(0, a) : b] = True
        return out


def decide(manifest: dict, probs: np.ndarray) -> np.ndarray:
    d, pos = manifest["decision"], manifest["positive_index"]
    if d["kind"] == "hysteresis" and manifest["task_type"] == "frame":
        on = hysteresis(
            probs[:, pos],
            manifest["frame_hop_ms"] / 1000,
            d["threshold"],
            d["deactivation_offset"],
            d["min_on_ms"],
            d["min_off_ms"],
            d["smooth_alpha_per_32ms"],
        )
        return np.where(on, pos, manifest["negative_index"])
    if d.get("threshold") is not None and pos is not None:
        rest = probs.copy()
        rest[:, pos] = -1
        return np.where(probs[:, pos] >= d["threshold"], pos, rest.argmax(-1))
    return probs.argmax(-1)


def segments(frames: list[str], hop_ms: float) -> list[dict]:
    out, cur, start = [], None, 0
    for i, f in enumerate(list(frames) + [None]):
        if f != cur:
            if cur is not None:
                out.append(
                    {"label": cur, "start_s": round(start * hop_ms / 1000, 3), "end_s": round(i * hop_ms / 1000, 3)}
                )
            cur, start = f, i
    return out


OUTSIDE = "O"
BEGIN, INSIDE, TAG_SEP = "B", "I", "-"


def bio(kind: str, n: int) -> list[str]:
    return [f"{BEGIN}{TAG_SEP}{kind}"] + [f"{INSIDE}{TAG_SEP}{kind}"] * (n - 1) if n else []


def _tag(tag: str) -> tuple[str, str]:
    head, sep, kind = tag.partition(TAG_SEP)
    if sep and head in (BEGIN, INSIDE):
        return head, kind
    return ("", "") if tag == OUTSIDE else (INSIDE, tag)


def spans(tags: list[str]) -> list[tuple[str, int, int]]:
    out, kind, start = [], "", 0
    for i, t in enumerate([*tags, OUTSIDE]):
        head, k = _tag(t)
        if kind and (head != INSIDE or k != kind):
            out.append((kind, start, i))
            kind = ""
        if k and not kind:
            kind, start = k, i
    return out


def first_subtokens(word_ids: list[int | None]) -> list[int]:
    out, prev = [], None
    for j, w in enumerate(word_ids):
        if w is not None and w != prev:
            out.append(j)
        prev = w
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _label_probs(labels: list[str], p: np.ndarray) -> dict[str, float]:
    return dict(map(lambda lab, v: (lab, round(float(v), 5)), labels, p))


def _normalize_image(img: Any, mean: list[float], std: list[float]) -> np.ndarray:
    a = (np.asarray(img, np.float32) / 255 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return a.transpose(2, 0, 1)


class LogMel:
    def __init__(self, cfg: dict, assets: Path):
        self.n_fft, self.hop, self.win = cfg["n_fft"], cfg["hop"], cfg["win_length"]
        self.window = np.load(assets / "window.npy").astype(np.float32)
        self.fb = np.load(assets / "mel_fb.npy").astype(np.float32)
        self.left = cfg["left_pad"]

    def frames(self, wav: np.ndarray) -> np.ndarray:
        n = 1 + (len(wav) - self.n_fft) // self.hop if len(wav) >= self.n_fft else 0
        if n <= 0:
            return np.zeros((self.fb.shape[1], 0), np.float32)
        idx = np.arange(self.n_fft)[None] + self.hop * np.arange(n)[:, None]
        spec = np.abs(np.fft.rfft(wav[idx] * self.window, axis=-1)) ** 2
        return np.log(spec.astype(np.float32) @ self.fb + 1e-6).T

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        return self.frames(np.concatenate([np.zeros(self.left, np.float32), wav.astype(np.float32)]))


_OCR = None
_PDFIUM = threading.Lock()


def _ocr(img: Any) -> str:
    global _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""
    if _OCR is None:
        _OCR = RapidOCR()
    res, _ = _OCR(np.asarray(img.convert("RGB")))
    return " ".join(r[1] for r in (res or []))


def _pdf_pages(path: Path, fe: dict) -> list[tuple[str, Any]]:
    import pypdfium2 as pdfium

    pages = []
    with _PDFIUM:
        pdf = pdfium.PdfDocument(str(path))
        for i in range(min(len(pdf), fe["max_pages"])):
            pg = pdf[i]
            text = pg.get_textpage().get_text_range() or ""
            pages.append((text, pg.render(scale=fe["dpi"] / 72).to_pil().convert("RGB")))
        pdf.close()
    return pages


def _fill_ocr(path: Path, pages: list[tuple[str, Any]], fe: dict) -> None:
    cache = path.with_name(path.name + ".ocr.json")
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    for i, (t, img) in enumerate(pages):
        if fe["ocr"] == "always" or len(t.strip()) < 20:
            pages[i] = (cached.get(str(i)) if str(i) in cached else _ocr(img), img)


def read_document(path: str | Path, fe: dict) -> list[tuple[str, Any]]:
    from PIL import Image

    path = Path(path)
    pages = _pdf_pages(path, fe) if path.suffix.lower() == ".pdf" else [("", Image.open(path).convert("RGB"))]
    if fe["route"] != "visual" and fe["ocr"] != "never":
        _fill_ocr(path, pages, fe)
    s = fe["image_size"]
    return [(t, img.resize((s, s))) for t, img in pages]


class Specialist:
    def __init__(self, path: str | Path, threads: int = 1, model_file: str | None = None):
        import onnxruntime as ort

        self.dir = Path(path)
        self.m = json.loads((self.dir / "manifest.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        model_path = self.dir / (model_file or self.m.get("model_file", "model.onnx"))
        self.sess = ort.InferenceSession(str(model_path), so, providers=prov)
        stream_path = self.dir / "stream.onnx"
        self.stream_sess = ort.InferenceSession(str(stream_path), so, providers=prov) if stream_path.exists() else None
        fe = self.m["frontend"]
        self.tok = self._tokenizer(fe)
        self.mel = LogMel(fe, self.dir) if fe["type"] == "logmel" else None

    def _tokenizer(self, fe: dict) -> Any:
        if not (fe["type"] == "tokenizer" or (fe["type"] == "document" and fe["route"] != "visual")):
            return None
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        tok.enable_truncation(fe["max_len"])
        if fe["type"] == "document":
            tok.enable_padding()
        return tok

    @property
    def name(self) -> str:
        return self.m["name"]

    @property
    def labels(self) -> list[str]:
        return self.m["labels"]

    def _encode_text(self, x: Any) -> Any:
        if self.m["task_type"] == "frame":
            return self.tok.encode(str(x).split(), is_pretokenized=True)
        return self.tok.encode(x)

    def _text_inputs(self, x: Any, fe: dict) -> dict[str, np.ndarray]:
        enc = self._encode_text(x)
        feeds = {"input_ids": np.array([enc.ids], np.int64), "attention_mask": np.array([enc.attention_mask], np.int64)}
        if "token_type_ids" in fe.get("inputs", []):
            feeds["token_type_ids"] = np.array([enc.type_ids], np.int64)
        return feeds

    def _document_inputs(self, x: Any, fe: dict) -> dict[str, np.ndarray]:
        pages = x if isinstance(x, list) else read_document(x, fe)
        feeds: dict[str, np.ndarray] = {}
        if fe["route"] != "visual":
            encs = self.tok.encode_batch([t or "" for t, _ in pages])
            feeds["input_ids"] = np.array([e.ids for e in encs], np.int64)
            feeds["attention_mask"] = np.array([e.attention_mask for e in encs], np.int64)
            if "token_type_ids" in fe["inputs"]:
                feeds["token_type_ids"] = np.array([e.type_ids for e in encs], np.int64)
        if fe["route"] != "text":
            pixels = [_normalize_image(im, fe["mean"], fe["std"]) for _, im in pages]
            feeds["pixel_values"] = np.stack(pixels).astype(np.float32)
        return feeds

    def _image_inputs(self, x: Any, fe: dict) -> dict[str, np.ndarray]:
        from PIL import Image

        img = (Image.open(x) if isinstance(x, (str, Path)) else x).convert("RGB").resize((fe["size"], fe["size"]))
        return {"pixel_values": _normalize_image(img, fe["mean"], fe["std"])[None].astype(np.float32)}

    def _inputs(self, x: Any) -> dict[str, np.ndarray]:
        fe = self.m["frontend"]
        if fe["type"] == "tokenizer":
            return self._text_inputs(x, fe)
        if fe["type"] == "document":
            return self._document_inputs(x, fe)
        if fe["type"] == "logmel":
            return {"mel": self.mel(np.asarray(x, np.float32))[None]}
        if fe["type"] == "waveform":
            return {"waveform": np.asarray(x, np.float32)[None]}
        if fe["type"] == "image":
            return self._image_inputs(x, fe)
        raise ValueError(fe["type"])

    def _run(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        emb, logits = self.sess.run(["embedding", "logits"], self._inputs(x))
        return emb[0], _softmax(logits[0])

    def encode(self, x: Any) -> np.ndarray:
        return self._run(x)[0]

    def predict(self, x: Any) -> dict:
        _, p = self._run(x)
        if self.m["task_type"] == "frame" and self.m["modality"] == "document":
            idx = decide(self.m, p)
            return {"pages": [self.labels[i] for i in idx], "probs": [_label_probs(self.labels, pi) for pi in p]}
        if self.m["task_type"] == "frame" and self.m["modality"] == "text":
            return self._tag_words(x, p)
        if self.m["task_type"] == "frame":
            idx = decide(self.m, p)
            frames = [self.labels[i] for i in idx]
            return {
                "segments": segments(frames, self.m["frame_hop_ms"]),
                "frames": frames,
                "probs": p[:, self.m["positive_index"]].round(4).tolist(),
            }
        idx = decide(self.m, p[None])[0]
        return {"label": self.labels[idx], "probs": _label_probs(self.labels, p)}

    def _tag_words(self, x: Any, p: np.ndarray) -> dict:
        words = str(x).split()
        first = first_subtokens(self._encode_text(x).word_ids)
        tags = [self.labels[i] for i in p[first].argmax(-1)]
        found = [{"label": k, "text": " ".join(words[a:b]), "start": a, "end": b} for k, a, b in spans(tags)]
        return {"entities": found, "tags": tags, "words": words[: len(tags)]}

    def stream(self) -> StreamSession:
        if self.stream_sess is None:
            raise ValueError(f"{self.name} has no streaming graph (only causal audio backbones stream)")
        return StreamSession(self)


class StreamSession:
    def __init__(self, sp: Specialist):
        self.sp, self.m = sp, sp.m
        fe = self.m["frontend"]
        self.hop = fe["hop"]
        self.wav = np.zeros(fe["left_pad"], np.float32)
        self.pending = np.zeros(0, np.float32)
        self.state = {
            i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32)
            for i in sp.stream_sess.get_inputs()
            if i.name != "mel"
        }
        d = self.m["decision"]
        self.vad = HysteresisState(
            self.m["frame_hop_ms"] / 1000,
            d["threshold"] if d["threshold"] is not None else 0.5,
            d["deactivation_offset"],
            d["min_on_ms"],
            d["min_off_ms"],
            d["smooth_alpha_per_32ms"],
        )
        self.t = 0

    def _positive_probs(self, buf: np.ndarray) -> np.ndarray:
        mel = self.sp.mel.frames(buf)[None]
        outs = self.sp.stream_sess.run(None, {"mel": mel, **self.state})
        names = [o.name for o in self.sp.stream_sess.get_outputs()]
        res = dict(map(lambda n, o: (n, o), names, outs))
        for n in self.state:
            self.state[n] = res[n + "_out"]
        return _softmax(res["logits"][0])[:, self.m["positive_index"]]

    def step(self, chunk: np.ndarray) -> dict:
        x = np.concatenate([self.pending, np.asarray(chunk, np.float32)])
        k = len(x) // self.hop
        self.pending = x[k * self.hop :]
        if k == 0:
            return {"probs": [], "speaking": self.vad.speaking, "events": []}
        buf = np.concatenate([self.wav, x[: k * self.hop]])
        self.wav = buf[-self.sp.mel.left :]
        p = self._positive_probs(buf)
        events = []
        for pi in p:
            ev = self.vad.push(float(pi))
            if ev:
                events.append({"type": ev, "t_s": round(self.t * self.m["frame_hop_ms"] / 1000, 3)})
            self.t += 1
        return {"probs": p.round(4).tolist(), "speaking": self.vad.speaking, "events": events}


def load(path: str | Path, threads: int = 1) -> Specialist:
    return Specialist(path, threads)


def _read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    wav, _ = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(1)


def _prep(m: Specialist, body: dict) -> Any:
    import base64

    mod = m.m["modality"]
    if mod == "document" and "pdf_b64" in body:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(base64.b64decode(body["pdf_b64"]))
        return f.name
    if mod == "audio":
        if "pcm16_b64" in body:
            return np.frombuffer(base64.b64decode(body["pcm16_b64"]), np.int16).astype(np.float32) / 32768
        return _read_wav(body["input"])
    return body["input"]


def _model_cards(models: list[Specialist]) -> dict:
    keys = ("name", "modality", "task_type", "labels", "metrics")
    return {"models": [{k: m.m[k] for k in keys if k in m.m} for m in models]}


def _route(by_name: dict[str, Specialist], pool: Any, path: str, body: dict) -> tuple[int, Any]:
    parts = path.strip("/").split("/")
    if parts == ["v1", "predict"]:
        futs = {n: pool.submit(m.predict, _prep(m, body)) for n, m in by_name.items()}
        return 200, {n: f.result() for n, f in futs.items()}
    if len(parts) == 3 and parts[0] == "v1" and parts[1] in by_name:
        m = by_name[parts[1]]
        if parts[2] == "predict":
            return 200, m.predict(_prep(m, body))
        if parts[2] == "encode":
            return 200, {"embedding": m.encode(_prep(m, body)).tolist()}
    return 404, {"error": "not found"}


def _handler(models: list[Specialist], by_name: dict[str, Specialist], pool: Any) -> type:
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a: Any) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                return self._send(200, _model_cards(models))
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                self._send(*_route(by_name, pool, self.path, body))
            except Exception as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

    return H


def serve(models: list[Specialist], host: str = "127.0.0.1", port: int = 8800) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from http.server import ThreadingHTTPServer

    by_name = {m.name: m for m in models}
    pool = ThreadPoolExecutor(max_workers=max(2, len(models)))
    print(f"byosm runtime: serving {', '.join(by_name)} on http://{host}:{port}")
    ThreadingHTTPServer((host, port), _handler(models, by_name, pool)).serve_forever()


def _parser() -> Any:
    import argparse

    ap = argparse.ArgumentParser(prog="byosm_rt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("model")
    p.add_argument("input")
    s = sub.add_parser("serve")
    s.add_argument("models", nargs="+")
    s.add_argument("--port", type=int, default=8800)
    s.add_argument("--host", default="127.0.0.1")
    return ap


def _predict_cli(model: str, inp: str) -> None:
    m = load(model)
    x: Any = _read_wav(inp) if m.m["modality"] == "audio" else inp
    out = m.predict(x)
    out.pop("frames", None)
    if m.m["task_type"] == "frame":
        out.pop("probs", None)
    print(json.dumps(out, indent=2))


def _main(argv: list[str]) -> None:
    a = _parser().parse_args(argv)
    if a.cmd == "predict":
        _predict_cli(a.model, a.input)
    else:
        serve([load(p) for p in a.models], a.host, a.port)


if __name__ == "__main__":
    _main(sys.argv[1:])
