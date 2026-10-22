from __future__ import annotations

from collections.abc import Callable

from .data.store import Dataset
from .data.sufficiency import MIN_CALIB_ITEMS, MIN_TEST_ITEMS
from .evals.selection import selection_value
from .project import Project

CALIB_SPLIT = "dev"


def held_out_size(p: Project, exp: str | None) -> tuple[int, int]:
    ds = Dataset(p.data_dir_for(p.exp_spec(exp) if exp else p.spec()))
    return len(ds.load("test")), len(ds.load(CALIB_SPLIT))


def bigger_held_out(p: Project) -> tuple[str, int, int] | None:
    own = p.spec().data_key()
    sizes = {e: held_out_size(p, e) for e in p.experiments() if p.exp_spec(e).data_key() != own}
    fit = {e: s for e, s in sizes.items() if s[0] >= MIN_TEST_ITEMS and s[1] >= MIN_CALIB_ITEMS}
    if not fit:
        return None
    exp = max(fit, key=lambda e: fit[e])
    return exp, *fit[exp]


def adopt_data(name: str, exp: str, log: Callable[[str], None] = print) -> str | None:
    from . import evals

    p = Project(name)
    spec = p.spec()
    before = held_out_size(p, None)
    spec.data = p.exp_spec(exp).data
    p.save_spec(spec)
    log(f"benchmark: {exp}'s data, {held_out_size(p, None)[0]} test rows (was {before[0]}); rescoring every model")
    board = evals.run(name, log=log)
    prim = spec.primary_suite()
    scored = {e: selection_value(board[e][prim]) for e in p.experiments() if prim in board.get(e, {})}
    if not scored:
        return None
    best = max(scored, key=lambda e: scored[e])
    p.promote(best)
    log(f"promoted {best}: calib {scored[best]:.4f} on {prim}, the best of {len(scored)} experiments")
    return best
