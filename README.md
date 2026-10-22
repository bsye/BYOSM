<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>

<h1 align="center">BYOSM</h1>

<p align="center"><b>Build Your Own Specialist Model.</b><br>One sentence in, a small model out that beats the big ones at its job, on a public benchmark, on any CPU.</p>

<p align="center">
  <a href="LICENSE"><img alt="Licence CC BY-NC 4.0" src="https://img.shields.io/badge/licence-CC%20BY--NC%204.0-lightgrey"></a>
  <a href="https://huggingface.co/bsye"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/models-Hugging%20Face-yellow"></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="Runs on CPU with ONNX" src="https://img.shields.io/badge/runs%20on-CPU%20%C2%B7%20ONNX-green">
</p>

<p align="center"><img src="docs/screenshots/hero.gif" alt="The specialists, then one specialist with its health, board, experiments, data and test sets" width="100%"></p>

## What it does

<table>
<tr>
<td width="33%" valign="top"><b>Bring your own LLM</b><br>Claude Code or Codex subscription, Ollama or LM Studio, or an Anthropic or OpenAI key. A model and effort per task, a token budget per job.</td>
<td width="33%" valign="top"><b>Finds data on Hugging Face</b><br>Searches the Hub and the web for public benchmarks and labelled datasets that fit your labels.</td>
<td width="33%" valign="top"><b>Writes data, then checks it</b><br>An LLM writes and labels training rows; a blind judge LLM relabels them and only rows it agrees with are kept.</td>
</tr>
<tr>
<td valign="top"><b>No leaks between splits</b><br>Near-duplicates are grouped before splitting. A data check flags thin labels and suggests datasets to add.</td>
<td valign="top"><b>Same scorer for every rival</b><br>LLMs, Hub models (with a measured label mapping), LiveKit and Silero, on the same frozen splits, with 95% intervals.</td>
<td valign="top"><b>Health score</b><br>Every check behind one score, a production-ready verdict, and what blocks it.</td>
</tr>
<tr>
<td valign="top"><b>Experiments the LLM proposes</b><br>Each with its reason and spec edits. The autopilot runs them and keeps a model only if it wins on calib.</td>
<td valign="top"><b>Every job live</b><br>Stages, progress, the LLM's decisions and the log as they happen. Tokens and calls per task and model.</td>
<td valign="top"><b>Your own test sets</b><br>Written by an LLM or sampled from the benchmark, scored by the models you pick, exported to Excel.</td>
</tr>
<tr>
<td valign="top"><b>Ships as int8 ONNX</b><br>One bundle that needs only numpy and onnxruntime. Python, command line, HTTP server or Docker image.</td>
<td valign="top"><b>Publishes to the Hub</b><br>With a model card that keeps what code measured apart from what an LLM wrote.</td>
<td valign="top"><b>Teams that act as you type</b><br>Specialists chained into one system that fires before the sentence ends, with no LLM at runtime.</td>
</tr>
</table>

## Run it

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
.venv/bin/byosm app                     # opens http://127.0.0.1:8765 with its token
```

State lives in one database: SQLite by default, Postgres with `DATABASE_URL` (`pip install -e '.[postgres]'`).

## Why the numbers hold up

A small model "beating GPT" is easy to fake. Each of these guards runs in code on every specialist; none is a promise in prose.

| Risk | Guard |
|---|---|
| Grading on your own homework | Rivals (LLMs, Hub models, LiveKit, Silero) go through the same scorer, on the same frozen splits, with thresholds calibrated on the same calib split. |
| Picking the lucky run | Promotion and the autopilot decide on calib only. Test is reported once, with a 95% bootstrap interval. |
| A lead that is noise | Every "ours vs theirs" is a paired bootstrap over the same rows; the health check counts a lead only when 95% of resamples keep it. |
| Memorising the training data | The overfitting check compares train, dev, calib and test, flags a dev curve that peaks and falls, and flags calib over-use. |
| Test rows leaking into training | Near-duplicate rows are grouped before splitting, so paraphrases never straddle train and test; the overfitting check also measures train/test overlap and the score without it. |
| Tiny test sets | Health fails "Precise score" below 100 test rows; a build adopts a bigger held-out set when an experiment brings one. |
| Only good on its own kind of data | "Holds on new inputs" scores it on test sets written separately; a model is not production ready until that is measured. |
| A metric that hides the trade-off | Constrained metrics carry their bound ("recall at 98% precision"), and a rival that never reaches the bound shows what it does reach. |
| Numbers typed by hand | The result tables are regenerated from the results database by `byosm report`; model cards separate what code measured from what an LLM wrote. |

Rules and benchmarks: [Methodology](docs/methodology.md). Every number: [Results](docs/results.md).

## Tour

| Step | What you do |
|---|---|
| [1. Describe the task](#1-describe-the-task) | One sentence, and the LLM you have |
| [2. Watch it build](#2-watch-it-build) | Follow each job live |
| [3. Check its health](#3-check-its-health) | See what fails and what to try next |
| [4. Compare with rivals](#4-compare-with-rivals) | Every model on the same splits |
| [5. Test it on new inputs](#5-test-it-on-new-inputs) | Your test sets and Hub rivals |
| [6. Act while the user types](#6-act-while-the-user-types) | A team fires before the sentence ends |
| [7. Choose the LLMs](#7-choose-the-llms) | What each task uses and costs |
| [8. Ship it](#8-ship-it) | Python, HTTP, Docker or the Hub |

### 1. Describe the task

One sentence and the LLM you have; the autopilot does the rest.

<img src="docs/screenshots/create.gif" alt="Typing a task and picking an LLM" width="100%">

### 2. Watch it build

Every job live: its stages, the LLM calls behind each, progress and the log.

<img src="docs/screenshots/job.gif" alt="A running job with its stages, progress and log" width="100%">

### 3. Check its health

A health score with every check behind it, and the experiments the LLM proposes to fix what fails.

<img src="docs/screenshots/health.gif" alt="Health checks and the proposed experiments" width="100%">

### 4. Compare with rivals

Test and calib scores with 95% intervals for every experiment and rival, scored by the same code.

<img src="docs/screenshots/board.gif" alt="A specialist's board against its rivals" width="100%">

### 5. Test it on new inputs

Hub models for the same task, ready to add as rivals, and test sets with every model's score.

<img src="docs/screenshots/test.gif" alt="Hugging Face rivals and a test set with each model's score" width="100%">

### 6. Act while the user types

A team of specialists picks the action and its arguments while the sentence is still being typed.

<img src="docs/screenshots/team.gif" alt="A team firing an action before the sentence ends" width="100%">

### 7. Choose the LLMs

Tokens and calls per task and model, and the model and effort each task uses.

<img src="docs/screenshots/models.gif" alt="LLM tokens and calls by task, and the model per task" width="100%">

### 8. Ship it

One ONNX bundle to call from Python, the command line, HTTP or a Docker image.

<img src="docs/screenshots/ship.gif" alt="How to use an exported specialist" width="100%">

Models on Hugging Face: [bsye](https://huggingface.co/bsye) ·
[byosm-doctype](https://huggingface.co/bsye/byosm-doctype) ·
[byosm-intent](https://huggingface.co/bsye/byosm-intent) ·
[byosm-vad](https://huggingface.co/bsye/byosm-vad)

<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>
