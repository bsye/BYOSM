from __future__ import annotations

from ..auto.standing import standing
from ..config import Project
from ..evals import board_rows
from ..format import cpu, params
from ..hub.facts import lineage, suite_dataset
from ..spec import TaskSpec
from .cells import STALE, fold_note, folded, num, score, stale_ids, stale_note, train_rows
from .origin import built_from
from .seeds import SEEDS, recipes, scored

NO_BASELINE = "no outside baseline scored yet"
SUMMARY_HEAD = (
    "| task | benchmark (metric) | our model | our test score [95% CI] | best outside baseline (on calib) "
    "| its test score | ours minus baseline on test (paired bootstrap) | calib, the selection split: ours vs baseline |"
    "\n|---|---|---|---|---|---|---|---|"
)


def _replicates(p: Project, prim: str) -> set[str]:
    line = {step["exp"] for step in lineage(p, p.promoted)} if p.promoted else set()
    return {i for ids in recipes(scored(p, prim)).values() for i in ids} - line


def _row(p: Project, r: dict, prim: str, stale: set[str]) -> str:
    mark = STALE if r["id"] in stale else ""
    name = f"**{r['id']}**{mark} (promoted)" if r["promoted"] else f"{r['id']}{mark}"
    exp = r["kind"] == "experiment"
    built = built_from(p, r["id"]) if exp else "outside baseline"
    cells = [name, built, train_rows(p, r["id"]) if exp else "-", params(r.get("params")), cpu(r)]
    cells += [score(r.get(prim), r.get(f"{prim}:ci")), num(r.get(f"{prim}:calib"))]
    return "| " + " | ".join(cells) + " |"


def _listed(p: Project, hidden: int) -> str:
    if not hidden:
        return "Every scored experiment is listed."
    return (
        f"Every scored experiment is listed, except {hidden} seed replicates off the promoted lineage, which are "
        f"summarised in the recipe table (`{p.name}:{SEEDS}`)."
    )


def _footer(p: Project, spec: TaskSpec, prim: str, notes: tuple[set[str], int, bool]) -> str:
    stale, hidden, any_folded = notes
    r = p.board().get(p.promoted or "", {}).get(prim, {})
    unit = "recordings" if spec.task_type == "frame" else "items"
    sizes = f"{r.get('n_test', '-')} test and {r.get('n_calib', '-')} calib {unit}"
    text = (
        f"Suite `{prim}` ({suite_dataset(spec, prim)[0]}): {sizes}. Selection uses calib; test is only reported. "
        f"{_listed(p, hidden)} CPU is PyTorch time per input, or per streaming chunk, timed when the row "
        "was last scored, so rows scored in different sessions are not a controlled speed comparison."
    )
    return text + fold_note(any_folded) + stale_note(stale)


def specialist_block(name: str) -> str:
    p = Project(name)
    spec = TaskSpec.load(p.spec_path)
    prim = spec.primary_suite()
    _, rows = board_rows(name)
    hidden = _replicates(p, prim)
    shown = [r for r in rows if r["id"] not in hidden]
    stale = stale_ids(p, prim)
    any_folded = any(r["kind"] == "experiment" and folded(p, r["id"]) for r in shown)
    metric = spec.objective.metric
    head = (
        f"| model | built from | train rows | params | CPU | test {metric} [95% CI] | calib {metric} |\n"
        "|---|---|---|---|---|---|---|"
    )
    body = "\n".join(_row(p, r, prim, stale) for r in shown)
    return f"{head}\n{body}\n\n{_footer(p, spec, prim, (stale, len(hidden), any_folded))}"


def _paired(d: dict | None) -> str:
    return "-" if d is None else f"{d['delta']:+.3f}, P(better) = {d['p_better']:.2f}"


def _calib(ours: float | None, theirs: float) -> str:
    if ours is None:
        return "-"
    word = "ahead" if ours > theirs else ("tied" if ours == theirs else "behind")
    return f"{num(ours)} vs {num(theirs)} ({word})"


def summary_row(name: str) -> str:
    p = Project(name)
    spec = TaskSpec.load(p.spec_path)
    prim = spec.primary_suite()
    board = p.board()
    ours = board.get(p.promoted or "", {}).get(prim)
    if ours is None:
        return f"| {spec.summary} | - | not promoted yet | - | - | - | - | - |"
    bench = f"{suite_dataset(spec, prim)[0]} ({spec.objective.metric})"
    model = f"{p.promoted} ({params(ours.get('params'))})"
    cells = [spec.summary, bench, model, score(ours["objective"], ours["metrics"].get("ci95"))]
    st = standing(name, 0.0)
    if st["best_baseline"] is None:
        cells += [NO_BASELINE, "-", "-", "-"]
    else:
        pid, theirs_calib = st["best_baseline"]
        theirs = board[pid][prim]
        cells += [
            f"{pid} ({params(theirs.get('params'))})",
            score(theirs["objective"], theirs["metrics"].get("ci95")),
            _paired(st["test"]["vs_best_baseline"]),
            _calib(st["ours"], theirs_calib),
        ]
    return "| " + " | ".join(cells) + " |"
