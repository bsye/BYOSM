from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import Predictor, register_baseline

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_KEY_ENV = "JEV_API_KEY"
JEV_WORKERS = 6
JEV_TIMEOUT_S = 60
JEV_RETRIES = 5
JEV_BACKOFF_S = 2.0
RETRY_CODES = (429, 500, 502, 503, 504)
QUESTION = "label"


def jev_identity() -> str:
    return f"{JEV_URL} {JEV_MODEL}"


@register_baseline("jev", "TypeSafe's Jev judgment model, given the same label definitions, one request per item.")
class Jev(Predictor):
    modalities, task_types = ("text",), ("classify",)

    def __init__(self, spec: TaskSpec):
        super().__init__(spec)
        from ..config import secret

        self.key = secret(JEV_KEY_ENV)
        if not self.key:
            raise ValueError(f"set {JEV_KEY_ENV} in the environment or the workspace .env to score jev")
        self.name = JEV_MODEL

    def _body(self, text: str) -> bytes:
        allowed = set(self.label_space or self.spec.label_names)
        criteria = {lab.name: lab.definition for lab in self.spec.labels if lab.name in allowed}
        question = {"type": "choice", "instructions": self.spec.description, "criteria": criteria}
        return json.dumps({"model": JEV_MODEL, "state": text, "questions": {QUESTION: question}}).encode()

    def _ask(self, text: str) -> dict[str, float]:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        req = urllib.request.Request(JEV_URL, data=self._body(text), headers=headers)
        for attempt in range(JEV_RETRIES):
            try:
                got = json.loads(urllib.request.urlopen(req, timeout=JEV_TIMEOUT_S).read())
                self.name = got.get("model", self.name)
                return got["answers"][QUESTION]["probabilities"]
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_CODES or attempt == JEV_RETRIES - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == JEV_RETRIES - 1:
                    raise
            time.sleep(JEV_BACKOFF_S * 2**attempt)
        return {}

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        from ..llm import pmap

        names = self.spec.label_names
        out = np.zeros((len(raws), len(names)), np.float32)
        for i, probs in enumerate(pmap(lambda x: self._ask(str(x)), raws, JEV_WORKERS)):
            for label, p in probs.items():
                if label in names:
                    out[i, names.index(label)] = p
        return out
