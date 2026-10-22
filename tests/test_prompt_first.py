from typer.testing import CliRunner

from byosm.cli import app
from byosm.config import Project
from byosm.spec import compile_spec


def test_the_compiler_names_the_task_and_avoids_taken_names(ws):
    first = compile_spec(None, "detect questions")
    assert first.name == "question-detector"
    Project(first.name).ensure()
    first.save(Project(first.name).spec_path)
    assert compile_spec(None, "detect questions").name == "question-detector-2"


def test_a_bare_prompt_runs_the_autopilot(ws):
    out = CliRunner().invoke(app, ["detect questions in chat", "--help"]).output
    assert "the task in plain language" in out


def test_a_new_spec_carries_a_title_for_people_and_when_it_was_created(ws):
    from datetime import datetime, timedelta

    spec = compile_spec(None, "detect questions")
    assert spec.title == "Question detector"
    assert datetime.now().astimezone() - spec.created < timedelta(minutes=1)
