from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from typer.core import TyperGroup

from ..project import Project

if TYPE_CHECKING:
    from ..spec import TaskSpec

HELP = """byosm - tiny System-1 specialists, from a sentence to a local model.

  auto       byosm "classify business documents into invoice, letter, ..."   -> everything below, decided automatically
             (name, modality and task type are inferred; `byosm auto <name>` resumes)
  define     byosm new vad "voice activity detection for a voice agent..."      -> spec.yaml (labels, objective, data plan)
             byosm suite add vad ava --hf nccratliri/vad-human-ava-speech --format segment_files
             byosm baseline add vad silero
  train      byosm build vad                     -> data (cached) -> LeJEPA -> train -> eval every suite -> promote
  compare    byosm board vad                     -> experiments vs baselines on every suite
  check      byosm overfit vad                   -> is it overfitting? train vs test gap, leakage, calib over-use
  improve    byosm improve vad                   -> diagnoses + proposals;  --apply a,b --run  -> next experiment
  ship       byosm export vad                    -> export/vad: ONNX + manifest + byosm_rt.py (numpy+onnxruntime only)
             byosm publish vad --repo you/byosm-vad   -> Hugging Face: bundle + model card (eval, comparison, lineage)
             byosm serve vad eot --port 8800     -> local HTTP API over exported bundles
"""

DEFAULT_COMMAND = "auto"


class PromptFirst(TyperGroup):
    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(cls=PromptFirst, add_completion=False, no_args_is_help=True, help=HELP)
suite_app = typer.Typer(no_args_is_help=True, help="Manage eval suites (frozen test sets) of a specialist.")
baseline_app = typer.Typer(no_args_is_help=True, help="Manage baselines (competitors scored on every suite).")
system_app = typer.Typer(no_args_is_help=True, help="Teams of specialists that do a whole task, scored end to end.")
inputs_app = typer.Typer(
    no_args_is_help=True, help="Sets of test inputs you write or sample, scored by the models you pick."
)
db_app = typer.Typer(no_args_is_help=True, help="The database that holds every specialist, result and report.")
exp_app = typer.Typer(no_args_is_help=True, help="Create experiment variants (ablations) by editing spec paths.")
lab_app = typer.Typer(
    no_args_is_help=True, help="Training methods an LLM writes as code, run only once a person approves them."
)
app.add_typer(lab_app, name="lab")
app.add_typer(suite_app, name="suite")
app.add_typer(baseline_app, name="baseline")
app.add_typer(exp_app, name="exp")
app.add_typer(db_app, name="db")
store_app = typer.Typer(
    no_args_is_help=True, help="Weights and exports in a private Hugging Face repo (artifact_repo)."
)
app.add_typer(store_app, name="store")
app.add_typer(system_app, name="system")
app.add_typer(inputs_app, name="inputs")
console = Console()


def log(msg: str) -> None:
    console.print(msg, highlight=False, markup=False)


def _spec(name: str) -> tuple[Project, TaskSpec]:

    p = Project(name)
    if not p.exists():
        raise typer.BadParameter(f"no specialist {name!r}")
    return p, p.spec()
