from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..modalities.audio import TinyCRNN
from ..runtime import Specialist

TOKEN_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


def _save_tokenizer(mod: Any, out: Path) -> None:
    mod.tokenizer.save_pretrained(out / "_tok")
    shutil.copy(out / "_tok" / "tokenizer.json", out / "tokenizer.json")
    shutil.rmtree(out / "_tok")


def _document(sp: Specialist, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    mod, dc = sp.mod, sp.spec.document
    names, example, dyn = [], [], {}
    if dc.route in ("text", "fused"):
        _save_tokenizer(mod, out)
        n_ex = min(2, dc.max_pages)
        enc = mod.tokenizer(["page one text", "page two"][:n_ex], padding=True, return_tensors="pt")
        for k in TOKEN_INPUTS:
            if k in enc:
                names.append(k)
                example.append(enc[k])
                dyn[k] = {0: "pages", 1: "seq"}
    if dc.route in ("visual", "fused"):
        names.append("pixel_values")
        example.append(torch.randn(min(2, dc.max_pages), 3, dc.image_size, dc.image_size))
        dyn["pixel_values"] = {0: "pages"}
    frontend = {
        "type": "document",
        "route": dc.route,
        "inputs": names,
        "max_pages": dc.max_pages,
        "max_len": dc.max_len,
        "image_size": dc.image_size,
        "dpi": dc.dpi,
        "ocr": dc.ocr,
        "mean": list(mod.norm[0]),
        "std": list(mod.norm[1]),
    }
    return frontend, names, tuple(example), dyn


def _text(sp: Specialist, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    mod = sp.mod
    _save_tokenizer(mod, out)
    enc = mod.tokenizer(["example input for export tracing"], return_tensors="pt")
    names = [k for k in TOKEN_INPUTS if k in enc]
    example = tuple(enc[k] for k in names)
    frontend = {"type": "tokenizer", "max_len": sp.spec.train.max_len, "inputs": names}
    return frontend, names, example, {k: {0: "batch", 1: "seq"} for k in names}


def _logmel(sp: Specialist, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    bb = sp.model.backbone
    mel = bb.mel
    np.save(out / "mel_fb.npy", mel.mel_scale.fb.numpy())
    np.save(out / "window.npy", mel.spectrogram.window.numpy())
    frontend = {
        "type": "logmel",
        "sample_rate": sp.spec.data.audio.sample_rate,
        "n_fft": mel.n_fft,
        "win_length": mel.win_length,
        "hop": bb.hop,
        "left_pad": 400 - bb.hop,
    }
    example = (torch.log(mel(torch.randn(1, 16000) * 0.1) + 1e-6),)
    return frontend, ["mel"], example, {"mel": {0: "batch", 2: "frames"}}


def _waveform(sp: Specialist, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    frontend = {"type": "waveform", "sample_rate": sp.spec.data.audio.sample_rate}
    return frontend, ["waveform"], (torch.randn(1, 16000) * 0.1,), {"waveform": {0: "batch", 1: "samples"}}


def _image(sp: Specialist, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    mod = sp.mod
    frontend = {"type": "image", "size": mod.size, "mean": list(mod.norm[0]), "std": list(mod.norm[1])}
    return frontend, ["pixel_values"], (torch.randn(1, 3, mod.size, mod.size),), {"pixel_values": {0: "batch"}}


def frontend_kind(sp: Specialist) -> str:
    if sp.spec.modality in ("document", "text"):
        return sp.spec.modality
    if isinstance(sp.model.backbone, TinyCRNN):
        return "logmel"
    if sp.spec.modality == "audio":
        return "waveform"
    return "image"


BUILDERS = {"document": _document, "text": _text, "logmel": _logmel, "waveform": _waveform, "image": _image}


def build_frontend(sp: Specialist, kind: str, out: Path) -> tuple[dict[str, Any], list[str], tuple, dict]:
    return BUILDERS[kind](sp, out)
