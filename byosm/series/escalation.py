from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .prediction import of
from .recipe import V1, Recipe
from .specialist import PLATEAU, promoted_version

Log = Callable[[str], None]


def recipe_of(report: dict[str, Any]) -> Recipe:
    return of(promoted_version(report)).fit.recipe


def tried(report: dict[str, Any], recipe: Recipe) -> bool:
    return any(of(v).fit.recipe == recipe for v in report.get("versions", []))


def upstream(recipe: Recipe = V1) -> dict[str, Any]:
    from .corpus import files
    from .patch import pretrained

    return {"base": (pretrained(recipe) or (None,))[0], "corpus": len(files())}


def lever(recipe: Recipe = V1) -> str:
    from .patch import missing_size

    size = missing_size(recipe)
    if not size:
        return "more pretraining data (series corpus --gb)"
    return f"the {size} base (series pretrain --size {size} --recipe {recipe.name})"


def _pretrain_missing(recipe: Recipe, why: str, log: Log) -> bool:
    from .corpus import PRETRAIN_STEPS
    from .patch import missing_size
    from .pretrain import pretrain

    size = missing_size(recipe)
    if size:
        log(f"  {why}: pretraining size {size} with recipe {recipe.name} first")
        pretrain(log, PRETRAIN_STEPS, size, recipe)
    return bool(size)


def escalate(report: dict[str, Any], log: Log) -> bool:
    why = f"more training stopped helping at effort {report[PLATEAU]['effort']}"
    if _pretrain_missing(recipe_of(report), why, log):
        return True
    log(
        f"  {why} and every base size is pretrained: "
        "the next lever is more pretraining data (series corpus --gb), so no new version is trained"
    )
    return False


def base_for(before: Recipe, after: Recipe, log: Log) -> None:
    from .patch import pretrained

    if after != before and not pretrained(after):
        _pretrain_missing(after, f"recipe {after.name} is new to this forecaster", log)
