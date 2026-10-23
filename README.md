<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>

<h1 align="center">BYOSM</h1>

<p align="center">
  <b>Build Your Own Specialist Model.</b><br>
  One sentence in, a small model for that one job, scored against LLMs, Jev and fine-tuned models on public benchmarks, on any CPU.
  <br><br>
  <a href="#run-it"><strong>Run it »</strong></a>
  &nbsp;·&nbsp;
  <a href="docs/results.md"><strong>Results »</strong></a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/bsye"><strong>Models on Hugging Face »</strong></a>
</p>

<p align="center">
  <a href="https://www.anthropic.com/claude-code"><img alt="Claude Code" src="https://img.shields.io/badge/Claude%20Code-D97757?style=for-the-badge&logo=claude&logoColor=white"></a>
  <a href="https://openai.com/codex"><img alt="Codex" src="https://img.shields.io/badge/Codex-000000?style=for-the-badge"></a>
  <a href="https://ollama.com"><img alt="Ollama" src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"></a>
  <a href="https://huggingface.co"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  <a href="https://onnx.ai"><img alt="ONNX" src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a>
  <a href="https://www.docker.com"><img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"></a>
  <a href="https://www.postgresql.org"><img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"></a>
</p>

<p align="center"><img src="docs/screenshots/hero.gif" alt="Home: your models against their rivals, their health and next actions, trending models and this machine, then one specialist with its board, experiments, data and test sets" width="100%"></p>

## Run it

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
.venv/bin/byosm app                     # opens http://127.0.0.1:8765
```

State lives in one database: SQLite by default, Postgres with `DATABASE_URL` (`pip install -e '.[postgres]'`).

## Tour

| Step | What you do |
|---|---|
| [1. Describe the task](#1-describe-the-task) | One sentence, and the LLM you have |
| [2. Watch it build](#2-watch-it-build) | Follow each job live |
| [3. Check its health](#3-check-its-health) | See what fails and what to try next |
| [4. Compare with rivals](#4-compare-with-rivals) | Every model on the same splits |
| [5. Test it on new inputs](#5-test-it-on-new-inputs) | Your test sets and Hugging Face rivals |
| [6. Act while the user types](#6-act-while-the-user-types) | A team fires before the sentence ends |
| [7. Choose the LLMs](#7-choose-the-llms) | What each task uses and costs |
| [8. Ship it](#8-ship-it) | Python, HTTP, Docker or Hugging Face |
| [9. Run it in a container](#9-run-it-in-a-container) | Each model or team isolated, live |
| [10. Know what it costs](#10-know-what-it-costs) | Build cost against an LLM, and when it pays off |

### 1. Describe the task

One sentence and the LLM you have; the autopilot does the rest. Home shows every model's health, its next action and what this machine can train.

<img src="docs/screenshots/create.gif" alt="Typing a task and picking an LLM" width="100%">

### 2. Watch it build

Every job live: its stages, the LLM calls behind each, progress and the log.

<img src="docs/screenshots/job.gif" alt="A running job with its stages, progress and log" width="100%">

### 3. Check its health

A health score with every check behind it, and the next step for each failing check, such as a lower learning rate when it overfits.

<img src="docs/screenshots/health.gif" alt="Health checks and the proposed experiments" width="100%">

### 4. Compare with rivals

Test and calib scores with 95% intervals for every experiment and rival, scored by the same code.

<img src="docs/screenshots/board.gif" alt="A specialist's board against its rivals" width="100%">

### 5. Test it on new inputs

Hugging Face models for the same task, ready to add as rivals, and test sets with every model's score.

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

### 9. Run it in a container

Deploy a specialist or a team into its own read-only container with CPU and memory limits, watch it live and send it inputs.

<img src="docs/screenshots/deploy.gif" alt="A team running in a container answering a request" width="100%">

### 10. Know what it costs

What each model cost to build, the LLM's cost per item measured on its rival runs, and after how many items the fine-tune pays off.

<img src="docs/screenshots/costs.gif" alt="Build cost, cost per item and the break-even curve against an LLM" width="100%">

Models on Hugging Face: [bsye](https://huggingface.co/bsye) ·
[byosm-doctype](https://huggingface.co/bsye/byosm-doctype) ·
[byosm-intent](https://huggingface.co/bsye/byosm-intent) ·
[byosm-vad](https://huggingface.co/bsye/byosm-vad)

<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>
