# AGENTS.md

Rules for anyone, human or agent, changing this repository. They follow the owner's coding guidelines (the same ones
used in their other projects) and what they have asked for in past sessions. When a rule and a habit disagree, the rule
wins.

## What this project is for

BYOSM turns one sentence and one command into a small specialist model that beats general models, including LLMs, in
its domain, measured on public benchmarks. Every change should serve that thesis or the honesty of its evidence.

## Product

- **One entry point.** Everything BYOSM builds is a specialist (text classifier, tagger, forecaster, event risk,
  team), created from one sentence in the Home prompt bar or New specialist. The app decides the kind, finds the data
  and fills every setting. Never add a per-kind creation button or a form that asks for columns, horizons or splits.
  Kinds show only as filter tabs and the badge on a card.
- **Best in class against a reference, automatically.** The goal of every build is a model comparable to the best
  public reference for its task. The path that got one specialist there (finding the reference, protocol checks,
  balancing data, pretraining, escalating Keep training) is the default path for the next one.
- **Experiments end as code.** Every experiment that works is consolidated into the pipeline, tested, and noted in
  `docs/design-notes.md`, so the app does it by default from then on. A result reached only by a scratch script, a
  manual setting or a one-off command does not count; land it in the code or drop it.
- **A feature closes the loop: feature, health, next steps, LLM.** Every feature that measures or improves a
  specialist adds a health check (or a finding in one) whose `args` are the step that fixes it, so it shows in Health
  and becomes a next step the ready loop and Measure everything run. A technique also gets a catalogue entry in
  `byosm/improve/techniques.yaml` naming the checks it `targets`. The proposer and lab LLMs read the health checks,
  the next steps and the catalogue with its targets, with one goal: raise the primary score and clear what is not ok.
- **Never cancel running work to free resources.** If jobs wait on the GPU, memory or a lock, fix the scheduling or
  the memory use; long trainings must be allowed to finish.

## Code

- **Never create a file with a shell redirect.** Use the file tool, which refuses to overwrite a file you have not
  read; a `cat >` once replaced `byosm/app/snapshot.py` and stopped every new job from starting.
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

- Every number in the docs, a model card or a commit comes from the database (`byosm db url`) and is
  regenerated, not typed by hand (`byosm report`).
- Benchmarks are public and industry-standard (AVA-Speech, Banking77, RVL-CDIP). Baselines are scored by the same code.
- Selection (promotion, autopilot keep/revert) decides on the calib split; test is only reported, with a 95% CI and a
  paired bootstrap against the baseline.
- Never claim what has not been run. If a feature exists but has not been measured, say so.
- Measure speed and accuracy before and after a change that could affect them.
- **Label definitions are what an LLM labels by.** They must state the edge cases the benchmark scores the same way
  the benchmark does; under a vague "fits" the judge relabelled 797 missing-argument rows the wrong way.
- A published reference score counts only after a protocol check: our run of a shared baseline (seasonal naive, base
  rate) must reproduce the published one on the same split. Compare with a reference only on the datasets both were
  scored on, and say on how many.

## UI

- The web app in `ui/` follows `docs/ui-guidelines.md`: stock shadcn components only, standard shadcn layout, clear
  hierarchy, equal-height cards in a grid, informative labels, every state designed.
- **Work in a cycle.** Build, screenshot the page, critique it against the guidelines, cut what does not earn its
  place, and repeat before committing.
- **Click by element, not by position.** In a browser check, find the element first and click its reference: a page
  that is still loading shifts, and a click by coordinates can land on a button that starts a job.
- **Formatting.** `ui/.prettierrc` is the style (no semicolons, 160 columns). Run `npx prettier --check` on the files
  you touch.
- No regex and no comments in TypeScript either.
- **Understandable at first glance.** Archetypal elements only: a list of cards that opens a detail page, the primary
  action top right, tabs for sections, stat cards, tables, a centred prompt bar. If a page needs a paragraph to be
  understood, change the page, not the paragraph.
- **Progressive, never blocking.** Anything that generates or scores shows results as they arrive (a job, or one
  request per item), with a spinner per pending item, and never disables other actions while it runs.
- **Cut text.** No description under a title unless the title alone is ambiguous. Labels are one to three words.
  Explanations go in a tooltip on an info icon, never as visible paragraphs. Every sentence on screen must change
  what the user does next; delete the rest.

## Docs

- README order: name and one-line pitch, a full-width app screenshot, how to run it, a screenshot tour with a one-line
  caption each. Nothing else: the goal and thesis live in `docs/assumptions.md`, methodology and benchmarks in
  `docs/methodology.md`, results in `docs/results.md`, the rest in `docs/details.md`.
- Every one of these pages starts and ends with the same contents line linking all of them.
- Retake the screenshots in `docs/screenshots/` when the UI changes, in light mode.
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
.venv/bin/byosm report                  # regenerate the result tables in docs/results.md
.venv/bin/byosm report --check          # fails if the tables in docs/results.md are stale
cd ui && npx tsc -b && npm run lint      # UI types and lint; `tsc -p .` checks nothing, the root tsconfig only references the app
```
