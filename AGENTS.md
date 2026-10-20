# AGENTS.md

Rules for anyone, human or agent, changing this repository. They follow the owner's coding guidelines (the same ones
used in their other projects) and what they have asked for in past sessions. When a rule and a habit disagree, the rule
wins.

## What this project is for

BYOSM turns one sentence and one command into a small specialist model that beats general models, including LLMs, in
its domain, measured on public benchmarks. Every change should serve that thesis or the honesty of its evidence.

## Code

- **No comments.** Names carry the meaning. The reasons behind a design go in `docs/design-notes.md`, not next to the
  code. The only docstrings allowed are the ones Typer prints as `--help` text for CLI commands.
- **No regex.** Use string methods (`split`, `partition`, `startswith`, `strip`, `str.isalnum`, ...).
- **Zero duplication, one source of truth.** Before writing a helper, search for an existing one. If two exist,
  merge them. A concept (a threshold, a label normaliser, a file path) lives in one named place that everything imports.
- **Small units.** Modules under 200 lines, functions short enough to read at a glance (split above ~40 lines).
  Exception: `byosm/rt.py` is the single-file runtime copied into every exported bundle and must stay one file that
  needs only `numpy` and `onnxruntime`.
- **Reduce lines of code.** Prefer the standard library and the libraries already in use over hand-rolled code.
- **No backwards-compatibility shims.** When something changes, migrate the data once and delete the old path.
- **Generic naming.** Name modules and functions for the concept, not for the first task that needed them.
- **Settings are configuration.** Tunable values live in `spec.yaml`, `byosm.yaml` or a named constant, never inline.
- **Cache expensive work** (OCR, embeddings, downloads) by content, so it is done once per machine.
- **Lint clean.** `ruff check` and `ruff format` pass. No `# noqa`: fix the cause or configure the rule in
  `pyproject.toml`.

## Evidence

- Every number in the README, a model card or a commit comes from a results file under `specialists/` and is
  regenerated, not typed by hand (`byosm report`).
- Benchmarks are public and industry-standard (AVA-Speech, Banking77, RVL-CDIP). Baselines are scored by the same code.
- Selection (promotion, autopilot keep/revert) decides on the calib split; test is only reported, with a 95% CI and a
  paired bootstrap against the baseline.
- Never claim what has not been run. If a feature exists but has not been measured, say so.
- Measure speed and accuracy before and after a change that could affect them.

## Docs

- README order: goal, then assumptions (the thesis), then results, then methodology and benchmarks, then details.
- Short and visual. Fewer words, more tables and diagrams.
- No em dashes.

## Git

- Never mention Claude, AI, an assistant or `Co-Authored-By` in commit messages, branch names or PR descriptions.
- Commit subjects are plain sentences that say what changed for a user of the project.
- Small, scoped commits: one concern each.
- Never commit data, model weights, exports, caches or secrets.

## Licensing and privacy

- Everything published (code, models, cards) is **CC BY-NC 4.0**.
- Never name a client or a person in code, docs, models or commits.

## Commands

```bash
.venv/bin/python -m pytest -q tests     # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/byosm report                  # regenerate the README result tables from specialists/*/board.json
.venv/bin/byosm report --check          # fails if the README result tables are stale
```
