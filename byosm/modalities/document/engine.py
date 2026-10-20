from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .store import STORE

_ocr_engine = None


def _engine(threads: int = -1) -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(intra_op_num_threads=threads) if threads > 0 else RapidOCR()


def ocr(img: Any) -> str:
    global _ocr_engine
    try:
        import numpy as np

        if _ocr_engine is None:
            _ocr_engine = _engine()
    except ImportError:
        return ""
    res, _ = _ocr_engine(np.asarray(img.convert("RGB")))
    return " ".join(r[1] for r in (res or []))


def _pool_init() -> None:
    global _ocr_engine
    _ocr_engine = _engine(threads=1)


def _ocr_store(job: tuple[Any, str | None]) -> str:
    img, key = job
    t = ocr(img)
    if key is not None:
        STORE.put(key, t)
    return t


def ocr_many(imgs: list[Any], keys: list[str] | None = None) -> list[str]:
    n = os.cpu_count() or 1
    jobs = list(zip(imgs, keys or [None] * len(imgs), strict=False))
    if len(imgs) < 2 * n or n == 1:
        return [_ocr_store(j) for j in jobs]
    try:
        importlib.import_module("rapidocr_onnxruntime")
    except ImportError:
        return [""] * len(imgs)
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_pool_init) as pool:
        return list(pool.map(_ocr_store, jobs, chunksize=4))
