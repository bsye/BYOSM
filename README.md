<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>

<h1 align="center">BYOSM</h1>

<p align="center"><b>Build Your Own Specialist Model.</b><br>One sentence in, a small model out that beats the big ones at its job, on a public benchmark, on any CPU.</p>

<p align="center">
  <a href="LICENSE"><img alt="Licence CC BY-NC 4.0" src="https://img.shields.io/badge/licence-CC%20BY--NC%204.0-lightgrey"></a>
  <a href="https://huggingface.co/bsye"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/models-Hugging%20Face-yellow"></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="Runs on CPU with ONNX" src="https://img.shields.io/badge/runs%20on-CPU%20%C2%B7%20ONNX-green">
</p>

<p align="center"><img src="docs/screenshots/specialists.jpg" alt="Every specialist against its best rival" width="100%"></p>
<p align="center"><sub>Every specialist, its lead over the best rival, and the jobs running now.</sub></p>

## Run it

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
.venv/bin/byosm app                     # opens http://127.0.0.1:8765 with its token
```

It brings its own LLM picker: local Ollama or LM Studio, Claude Code, Codex, or an API key.

## Tour

### Describe the task

<img src="docs/screenshots/create.jpg" alt="Create: one sentence and an LLM" width="100%">

One sentence and the LLM you have; the autopilot does the rest.

### Check its health

<img src="docs/screenshots/specialist.jpg" alt="A specialist with its health checks and next steps" width="100%">

A health score with every check behind it, and the next steps to fix what fails.

### Pick the next experiment

<img src="docs/screenshots/experiments.jpg" alt="Health and the proposed experiments side by side" width="100%">

Health and next steps: proposed experiments, each with why and the spec changes it makes.

### See it against its rivals

<img src="docs/screenshots/board.jpg" alt="A specialist's board against Jev and the LLM" width="100%">

Test and calib scores with 95% intervals for every model and rival, scored by the same code.

### Test it against Hugging Face

<img src="docs/screenshots/rivals.jpg" alt="Hugging Face models that can be added as rivals" width="100%">

Hub models for the same task, ready to add and score as rivals.

### Test it on new inputs

<img src="docs/screenshots/testsets.jpg" alt="A test set with every model's score" width="100%">

Named test sets written by an LLM or taken from the benchmark, with each model's score.

### Watch it build

<img src="docs/screenshots/job.jpg" alt="A running job with its stages and progress" width="100%">

Every job live: its stage, its progress and its log.

### Act while the user types

<img src="docs/screenshots/team.jpg" alt="A team firing an action after 33 of 55 characters" width="100%">

A team of specialists fires the action before the sentence ends, with no LLM at runtime.

### Know what the LLM costs

<img src="docs/screenshots/models.jpg" alt="LLM tokens and calls by task" width="100%">

Tokens and calls per task, and which LLM each running job uses.

Models on Hugging Face: [bsye](https://huggingface.co/bsye) ·
[byosm-doctype](https://huggingface.co/bsye/byosm-doctype) ·
[byosm-intent](https://huggingface.co/bsye/byosm-intent) ·
[byosm-vad](https://huggingface.co/bsye/byosm-vad)

<p align="center"><b>Tour</b> · <a href="docs/assumptions.md">Goal and bet</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/results.md">Results</a> · <a href="docs/details.md">Details</a></p>
