from __future__ import annotations

import io
from typing import Any

import numpy as np


def _decode_audio(v: Any, sr: int) -> np.ndarray:
    import soundfile as sf

    if isinstance(v, dict) and v.get("bytes"):
        wav, file_sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32", always_2d=True)
    elif isinstance(v, dict) and "array" in v:
        wav, file_sr = np.asarray(v["array"], dtype=np.float32)[:, None], v["sampling_rate"]
    else:
        wav, file_sr = sf.read(v["path"], dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if file_sr != sr:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return wav.astype(np.float32)


def _decode_image(v: Any) -> Any:
    from PIL import Image

    if isinstance(v, dict) and v.get("bytes"):
        return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    return Image.open(v["path"]).convert("RGB")


def _decode(v: Any, modality: str, sr: int) -> Any:
    if modality == "text":
        return str(v)
    if modality == "audio":
        return _decode_audio(v, sr)
    if modality in ("image", "document"):
        return _decode_image(v)
    raise ValueError(modality)
