"""Autopilot: one command + a prompt -> a specialist that beats the baselines (incl. LLMs) on its domain, or a clear
report of why it couldn't within budget.

  define    compile the spec from the prompt; discover a public benchmark + baseline models; add LLM baseline(s)
  build     v1
  loop      diagnose -> pick the most promising untried proposal -> build it -> keep if it improves the primary suite
            (one change per round, so every gain is attributable)
  stop      won (beats every baseline by `margin` and meets the target) | budget | plateau | nothing left to try
  distill   if the winner is large: distill into a tiny student; keep it if it still wins
  ship      export (+ publish)
Every decision and its evidence goes to specialists/<name>/auto.json and auto.md.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import Project, load_config

# Prior expected gain per proposal kind (higher first). Ties keep the rule order.
PRIORS = {"real-data": 3.0, "domain-encoder": 3.0, "utterance-labels": 2.5, "new-split": 2.0, "more-data": 2.0, "regularize": 1.5,
          "hard-pairs": 1.5, "llm-": 1.5, "train-longer": 1.0, "refine-more": 1.0}


def _prior(pid: str) -> float:
    return next((v for k, v in PRIORS.items() if pid.startswith(k)), 1.0)


def _key(pr: dict) -> str:
    return json.dumps(pr["ops"], sort_keys=True)


def standing(name: str, margin: float) -> dict[str, Any]:
    """Where the promoted experiment stands on the primary suite. Decisions use the calib split; test is reported."""
    from . import evals
    from .spec import TaskSpec

    p = Project(name)
    base = TaskSpec.load(p.spec_path)
    prim = base.primary_suite()
    board = p.board()
    mine = board.get(p.promoted or "", {}).get(prim)
    ours = evals.selection_value(mine)
    bls = {k: evals.selection_value(v[prim]) for k, v in board.items() if prim in v and v[prim]["kind"] == "baseline"}
    best_bl = max(bls.items(), key=lambda kv: kv[1]) if bls else None
    beats = ours is not None and (best_bl is None or ours > best_bl[1] + margin)
    meets = ours is not None and ours >= base.objective.target
    test = {"ours": mine.get("objective") if mine else None, "ci95": (mine or {}).get("metrics", {}).get("ci95"),
            "baselines": {k: v[prim]["objective"] for k, v in board.items() if prim in v and v[prim]["kind"] == "baseline"},
            "vs_best_baseline": evals.compare(base, mine, board[best_bl[0]][prim], "test") if best_bl and mine else None}
    return {"suite": prim, "ours": ours, "exp": p.promoted, "baselines": bls, "best_baseline": best_bl, "beats": beats, "meets": meets,
            "won": beats and meets, "metric": base.objective.metric, "target": base.objective.target, "test": test}


class Log:
    def __init__(self, p: Project, echo: Callable[[str], None]):
        self.p, self.echo, self.decisions = p, echo, []
        f = p.root / "auto.json"
        if f.exists():
            self.decisions = json.loads(f.read_text()).get("decisions", [])

    def __call__(self, step: str, choice: Any, why: str, **extra: Any) -> None:
        d = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "choice": choice, "why": why, **extra}
        self.decisions.append(d)
        self.echo(f"[auto] {step}: {choice} - {why}")
        self.save()

    def save(self, final: dict | None = None) -> None:
        self.p.root.mkdir(parents=True, exist_ok=True)
        (self.p.root / "auto.json").write_text(json.dumps({"decisions": self.decisions, "final": final}, indent=2, default=str))


def run(name: str, prompt: str | None = None, modality: str | None = None, budget_h: float = 6.0, max_experiments: int = 8,
        margin: float = 0.0, distill: bool = True, max_params_m: float = 25.0, publish: str | None = None, license: str = "other",
        use_llm: bool = True, echo: Callable[[str], None] = print) -> dict[str, Any]:
    from . import distill as dst
    from . import evals, improve
    from .discover import discover
    from .pipeline import build
    from .spec import TaskSpec, compile_spec

    p = Project(name)
    cfg = load_config(p.ws)
    log = Log(p, echo)
    t0 = time.time()
    deadline = t0 + budget_h * 3600

    # ---- define
    if not p.exists():
        if not prompt:
            raise ValueError(f"{name} doesn't exist yet: pass the task prompt")
        spec = compile_spec(name, prompt, modality, cfg)
        log("define", f"{spec.modality}/{spec.task_type}, {len(spec.labels)} labels, backbone {spec.backbone}", f"compiled from the prompt; objective {spec.objective.metric} >= {spec.objective.target}")
        spec, decisions = discover(spec, spec.search, cfg, echo)
        for d in decisions:
            log("discover:" + d["step"], d["choice"], d["why"])
        p.ensure()
        spec.save(p.spec_path)
    base = TaskSpec.load(p.spec_path)

    # ---- v1
    durations: list[float] = []
    if not p.experiments():
        ts = time.time()
        build(name, "v1", echo)
        durations.append(time.time() - ts)
        st = standing(name, margin)
        log("build", "v1", f"{st['suite']} {st['metric']} {st['ours']:.4f}; best baseline {st['best_baseline']}")

    # ---- improvement loop
    tried = {_key(d["proposal"]) for d in log.decisions if d["step"] == "experiment" and "proposal" in d}
    stale, reason = 0, ""
    while True:
        st = standing(name, margin)
        if st["won"]:
            reason = f"won: {st['exp']} {st['ours']:.4f} beats every baseline" + (f" (best {st['best_baseline'][0]} {st['best_baseline'][1]:.4f})" if st["best_baseline"] else "") + f" and meets target {st['target']}"
            break
        n_exp = len(p.experiments())
        if n_exp >= max_experiments:
            reason = f"experiment budget reached ({n_exp}/{max_experiments})"
            break
        est = (sum(durations) / len(durations)) if durations else 0
        if time.time() + est > deadline:
            reason = f"time budget: {(deadline - time.time()) / 3600:.1f}h left < ~{est / 3600:.1f}h per experiment"
            break
        dg, props = improve.propose(name, use_llm, echo)
        fresh = [pr for pr in props if _key(pr) not in tried]
        if not fresh:
            reason = "no untried proposals left"
            break
        left_min = (deadline - time.time()) / 60
        fits = [pr for pr in fresh if not (pr.get("cost", {}).get("minutes", 0) > left_min)]  # NaN (unknown) fits
        if not fits:
            reason = f"time budget: {left_min:.0f} min left, cheapest untried proposal needs ~{min(pr['cost']['minutes'] for pr in fresh):.0f} min"
            break
        # Highest prior first; among equals the cheaper one (gain per minute).
        pick = max(fits, key=lambda pr: (_prior(pr["id"]), -(pr.get("cost", {}).get("minutes") or 0), -fits.index(pr)))
        tried.add(_key(pick))
        before = st["ours"]
        exp = improve.apply(name, [pick["id"]])
        ts = time.time()
        try:
            build(name, exp, echo)
        except Exception as e:  # noqa: BLE001
            log("experiment", exp, f"{pick['id']} failed: {type(e).__name__}: {str(e)[:200]}", proposal=pick)
            stale += 1
            continue
        durations.append(time.time() - ts)
        after = standing(name, margin)
        board = p.board()
        res = board.get(exp, {}).get(st["suite"])
        mine = evals.selection_value(res)
        cmp = evals.compare(base, res, board.get(st["exp"] or "", {}).get(st["suite"]))
        kept = after["exp"] == exp
        # A gain counts only if it is unlikely to be split noise: P(better on calib) >= 0.9 under a paired bootstrap.
        real = kept and (cmp["p_better"] >= 0.9 if cmp else mine is not None and before is not None and mine - before > 0.002)
        log("experiment", exp, f"{pick['id']}: {st['suite']} calib {mine:.4f} vs {before:.4f}" + (f" (P(better)={cmp['p_better']:.2f})" if cmp else "")
            + f" -> {'kept (promoted)' if kept else 'reverted'}. Evidence: {pick['why']}",
            proposal=pick, delta=None if mine is None or before is None else round(mine - before, 4), p_better=cmp and cmp["p_better"],
            alternatives=[pr["id"] for pr in fresh if pr is not pick])
        stale = 0 if real else stale + 1
        if stale >= 2:
            reason = "plateau: 2 rounds without a significant gain"
            break
    log("stop", reason, "")

    # ---- distill
    st = standing(name, margin)
    if distill and st["exp"]:
        meta = json.loads((p.exp_dir(st["exp"]) / "meta.json").read_text())
        if meta["params"] / 1e6 > max_params_m and not meta.get("distilled_from"):
            if time.time() + (sum(durations) / max(1, len(durations))) <= deadline:
                teacher = st["exp"]
                exp = dst.create(name, teacher)
                build(name, exp, echo)
                board = p.board()
                stu = board.get(exp, {}).get(st["suite"], {}).get("objective")
                smeta = json.loads((p.exp_dir(exp) / "meta.json").read_text())
                still = stu is not None and (st["best_baseline"] is None or stu > st["best_baseline"][1] + margin)
                if still:
                    Project(name).promote(exp)
                log("distill", exp, f"{meta['params'] / 1e6:.1f}M teacher {teacher} {st['ours']:.4f} -> {smeta['params'] / 1e6:.1f}M student {stu:.4f}; "
                    + ("student still beats every baseline -> promoted" if still else "student no longer wins -> keep the teacher"))
            else:
                log("distill", None, "skipped: not enough time budget left")
        else:
            log("distill", None, f"not needed: {meta['params'] / 1e6:.1f}M <= {max_params_m}M" if not meta.get("distilled_from") else "already a distilled student")

    # ---- ship
    from .export import export

    out = export(name, log=echo)
    log("export", str(out.relative_to(p.ws)), "torch-free ONNX bundle (encode / predict / serve)")
    if publish:
        from .hub import publish as pub

        pub(name, publish, license, private=True, log=echo)
        log("publish", publish, "private Hugging Face repo with generated model card (flip public on the Hub)")
    final = standing(name, margin)
    final["hours"] = round((time.time() - t0) / 3600, 2)
    final["stop_reason"] = reason
    log.save(final)
    _report(p, log.decisions, final)
    return final


def _report(p: Project, decisions: list[dict], final: dict) -> None:
    from . import evals

    suites, rows = evals.board_rows(p.name)
    t = final.get("test") or {}
    ci = t.get("ci95")
    vs = t.get("vs_best_baseline")
    bb = final["best_baseline"]
    lines = [f"# {p.name}: autopilot report", "",
             f"**Result:** {final['exp']}: **{'WON' if final['won'] else 'did not win'}** ({final['stop_reason']}, {final['hours']} h).", "",
             f"- selection (`{final['suite']}` calib split): {final['ours']:.4f} {final['metric']}" + (f" vs best baseline {bb[0]} {bb[1]:.4f}" if bb else ""),
             f"- held-out test: {t['ours']:.4f}" + (f" (95% CI {ci['lo']:.4f}-{ci['hi']:.4f})" if ci else "")
             + (f" vs {bb[0]} {t['baselines'][bb[0]]:.4f}: delta {vs['delta']:+.4f}, P(better) {vs['p_better']:.2f}" if vs and bb else "") if t.get("ours") is not None else "- held-out test: n/a",
             "",
             "## Board", "", "| | params | " + " | ".join(suites) + " |", "|---|---|" + "---|" * len(suites)]
    for r in rows:
        lines.append(f"| {r['id']}{' *' if r['promoted'] else ''} | {(r['params'] or 0) / 1e6:.2f}M | " + " | ".join("–" if r.get(s) is None else f"{r[s]:.4f}" for s in suites) + " |")
    lines += ["", "## Decisions", ""]
    for d in decisions:
        lines.append(f"- `{d['t']}` **{d['step']}** → {d['choice']}: {d['why']}")
    (p.root / "auto.md").write_text("\n".join(lines) + "\n")
