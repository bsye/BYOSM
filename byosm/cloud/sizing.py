from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MEMORY_FACTOR = 2.0
RUNTIME_MB = 300
UTILIZATION = 0.6
DEFAULT_MS = 50.0
CPU_UNITS_PER_VCPU = 1024
FARGATE = {
    256: [512, 1024, 2048],
    512: list(range(1024, 4097, 1024)),
    1024: list(range(2048, 8193, 1024)),
    2048: list(range(4096, 16385, 1024)),
    4096: list(range(8192, 30721, 1024)),
}


def _latency_ms(manifest: dict[str, Any]) -> float:
    q = manifest.get("quantization") or {}
    tried = next(iter((q.get("tried") or {}).values()), None)
    chosen = tried if manifest.get("precision") == "int8" and tried else q.get("fp32")
    return float((chosen or {}).get("ms") or DEFAULT_MS)


def _manifests(bundle: Path) -> list[dict[str, Any]]:
    found = [bundle / "manifest.json", *bundle.glob("nodes/*/manifest.json")]
    return [json.loads(p.read_text()) for p in found if p.exists()]


def size(bundle: Path, rps: float) -> dict[str, Any]:
    model_mb = sum(p.stat().st_size for p in bundle.rglob("*.onnx")) / 1e6
    manifests = _manifests(bundle)
    latency = sum(_latency_ms(m) for m in manifests if "quantization" in m or "precision" in m) or DEFAULT_MS
    params = sum(int(m.get("params") or 0) for m in manifests)
    need_mb = model_mb * MEMORY_FACTOR + RUNTIME_MB
    need_cpu = rps * latency / 1000 / UTILIZATION * CPU_UNITS_PER_VCPU
    cpu = next(c for c in FARGATE if c >= need_cpu or c == max(FARGATE))
    while need_mb > FARGATE[cpu][-1] and cpu < max(FARGATE):
        cpu *= 2
    memory = next((m for m in FARGATE[cpu] if m >= need_mb), FARGATE[cpu][-1])
    return {
        "params": params,
        "model_mb": round(model_mb, 1),
        "latency_ms": round(latency, 2),
        "rps": rps,
        "need_mb": math.ceil(need_mb),
        "need_vcpu": round(need_cpu / CPU_UNITS_PER_VCPU, 3),
        "cpu": cpu,
        "memory": memory,
        "why": (
            f"memory: {model_mb:.0f} MB of ONNX x {MEMORY_FACTOR:g} + {RUNTIME_MB} MB runtime = {need_mb:.0f} MB; "
            f"cpu: {rps:g} requests/s x {latency:.1f} ms at {UTILIZATION:.0%} use = {need_cpu / CPU_UNITS_PER_VCPU:.2f} vCPU"
        ),
    }
