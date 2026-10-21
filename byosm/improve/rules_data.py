from __future__ import annotations

from typing import Any

from .registry import gap, rule


def _train_partition(source: Any) -> Any:
    if source.format == "segment_files":
        upd = {"role": "labeled", "partition": "train", "max_files": 60, "max_rows": 6000}
    else:
        upd = {"role": "labeled", "split": "train" if source.split != "train" else source.split}
    return source.model_copy(update=upd)


def _real_data_why(name: str, r: dict, bb: tuple[str, float] | None, dgap: float) -> str:
    vs = f" vs {bb[0]} {bb[1]:.3f}" if bb else ""
    below = f", {dgap:.3f} below builtin" if dgap > 0 else ""
    tail = ". The model never trained on this source; use its train partition (disjoint from the suite's calib/test)."
    return f"{name}: {r['objective']:.3f}" + vs + below + tail


@rule
def real_data(dg: dict) -> list[dict]:
    spec, base, out = dg["spec"], dg["base"], []
    trained = {s.dataset for s in spec.data.hf_sources if s.role == "labeled"}
    builtin = dg["suites"].get("builtin", {}).get("objective")
    for suite in base.evals:
        r = dg["suites"].get(suite.name)
        if not r or suite.kind != "hf" or not suite.source or suite.source.dataset in trained:
            continue
        lag, bb = gap(dg, suite.name)
        dgap = (builtin - r["objective"]) if builtin is not None else 0
        if lag > 0 or dgap > 0.05:
            src = _train_partition(suite.source)
            ops = [{"op": "append", "path": "data.hf_sources", "value": src.model_dump(mode="json")}]
            out.append({"id": f"real-data-{suite.name}", "ops": ops, "why": _real_data_why(suite.name, r, bb, dgap)})
    return out


def _curve_note(dg: dict) -> str:
    curve = dg["curve"]
    if len(curve) < 2:
        return ""
    tail = curve[-max(2, len(curve) // 3) :]
    if dg["late_best"]:
        return f"; dev was still rising at the end ({curve[0]:.2f} -> {max(curve):.2f}), consistent with being data-limited"
    if max(tail) - min(tail) < 0.01:
        return (
            f"; dev plateaued at {max(curve):.2f}, so more of the same data may help less than new data or a new recipe"
        )
    return f"; dev reached {max(curve):.2f}"


def _more_data_evidence(dg: dict, lag: float, bb: tuple[str, float] | None) -> str:
    spec, primary = dg["spec"], dg["primary"]
    obj = dg["suites"][primary]["objective"]
    gap_note = f"{lag:.3f} behind {bb[0]} ({bb[1]:.3f})" if lag > 0 else f"below target {spec.objective.target}"
    return f"{primary}: {obj:.3f}, " + gap_note + f", with {dg['n_train']} training examples" + _curve_note(dg)


def _more_from_source(dg: dict, i: int, s: Any, ev: str) -> list[dict]:
    if s.sampling == "balanced":
        return _more_balanced(dg, i, s, ev)
    if s.format == "segment_files":
        ops = [
            {"op": "set", "path": f"data.hf_sources.{i}.max_files", "value": s.max_files * 2},
            {"op": "set", "path": f"data.hf_sources.{i}.max_rows", "value": s.max_rows * 2},
        ]
        return [{"id": "more-data", "ops": ops, "why": ev + f". Take 2x more files/clips from {s.dataset}."}]
    return _more_stream(dg, i, s, ev)


def _more_stream(dg: dict, i: int, s: Any, ev: str) -> list[dict]:
    from ..data import hf_source

    size = hf_source.split_sizes(s.dataset, s.config).get(s.split)
    read = dg.get("read_by_source", {}).get(f"hf:{s.dataset}", 0)
    cap = sum(x.max_rows for x in dg["spec"].data.hf_sources if x.dataset == s.dataset)
    if size is not None and size <= s.max_rows or size is None and 0 < read < cap:
        alt = _unused_split(dg, hf_source, s)
        if not alt:
            return []
        used = f"all {size} rows read" if size is not None else f"it ran out at {read} rows, below the {cap} cap"
        why = ev + f". {s.split} is used up ({used}); `{alt[0]}` has {alt[1]} rows that no eval suite reads."
        value = s.model_copy(update={"split": alt[0], "max_rows": alt[1]}).model_dump(mode="json")
        return [{"id": "new-split", "ops": [{"op": "append", "path": "data.hf_sources", "value": value}], "why": why}]
    ops = [{"op": "set", "path": f"data.hf_sources.{i}.max_rows", "value": s.max_rows * 4}]
    return [{"id": "more-data", "ops": ops, "why": ev + f". Read 4x more rows from {s.dataset}."}]


def _richest_labeled(dg: dict) -> tuple[int, Any] | None:
    rows = dg.get("rows_by_source") or {}
    labeled = [(i, s) for i, s in enumerate(dg["spec"].data.hf_sources) if s.role == "labeled"]
    used = [(i, s) for i, s in labeled if rows.get(f"hf:{s.dataset}")]
    return max(used, key=lambda t: rows[f"hf:{t[1].dataset}"]) if used else next(iter(labeled), None)


@rule
def more_data(dg: dict) -> list[dict]:
    spec, out = dg["spec"], []
    lag, bb = gap(dg, dg["primary"])
    below = (dg["suites"].get(dg["primary"], {}).get("objective") or 0) < spec.objective.target
    if not (lag > 0 or below):
        return out
    ev = _more_data_evidence(dg, lag, bb)
    labeled = _richest_labeled(dg)
    if labeled:
        out += _more_from_source(dg, *labeled, ev)
    if not out and spec.data.synthetic:
        ops = [{"op": "set", "path": "data.train_size", "value": spec.data.train_size * 2}]
        out.append({"id": "more-data", "ops": ops, "why": ev + ". Generate 2x more training data."})
    return out


def _per_label(i: int, value: int, why: str) -> list[dict]:
    return [
        {
            "id": "more-data",
            "ops": [{"op": "set", "path": f"data.hf_sources.{i}.per_label", "value": value}],
            "why": why,
        }
    ]


def _availability(hf_source: Any, s: Any, spec: Any) -> int | None:
    try:
        return min(hf_source.availability(s, spec).values() or [0])
    except Exception:
        return None


def _unused_split(dg: dict, hf_source: Any, s: Any) -> tuple[str, int] | None:
    spec, base = dg["spec"], dg["base"]
    used = {e.source.split for e in base.evals if e.source and e.source.dataset == s.dataset}
    used |= {x.split for x in spec.data.hf_sources if x.dataset == s.dataset}
    sizes = hf_source.split_sizes(s.dataset, s.config)
    alt = max((k for k in sizes if k not in used), key=lambda k: sizes[k], default=None)
    return (alt, sizes[alt]) if alt else None


def _new_split(dg: dict, hf_source: Any, s: Any, have: int, want: int, ev: str) -> list[dict]:
    alt = _unused_split(dg, hf_source, s)
    n_labels = len(dg["spec"].labels)
    if not alt or alt[1] // n_labels <= s.per_label // 2:
        return []
    per = min(want, alt[1] // n_labels)
    value = s.model_copy(update={"split": alt[0], "per_label": per}).model_dump(mode="json")
    why = (
        ev + f". {s.split} is used up ({have} per label, {s.per_label} taken); "
        f"`{alt[0]}` has {alt[1]} rows that no eval suite reads: add {per} per label from it."
    )
    return [{"id": "new-split", "ops": [{"op": "append", "path": "data.hf_sources", "value": value}], "why": why}]


def _more_balanced(dg: dict, i: int, s: Any, ev: str) -> list[dict]:
    from ..data import hf_source

    want = s.per_label * 4
    have = _availability(hf_source, s, dg["spec"])
    if have is None or want <= have:
        return _per_label(i, want, ev + f". {s.dataset} ({s.split}) has more rows per label: {s.per_label} -> {want}.")
    if have >= s.per_label * 1.5:
        why = f". {s.dataset} ({s.split}) has {have} rows for its rarest label: {s.per_label} -> {have} per label (all of it)."
        return _per_label(i, have, ev + why)
    return _new_split(dg, hf_source, s, have, want, ev)
