<p align="center"><a href="../README.md">Tour</a> · <a href="assumptions.md">Goal and bet</a> · <a href="methodology.md">Methodology</a> · <a href="results.md">Results</a> · <b>Details</b></p>

# Details

## Features

| Build | Evaluate | Observe | Ship |
|---|---|---|---|
| ✅ Claude Code or Codex | ✅ Same scorer for all rivals | ✅ Every job live | ✅ int8 ONNX on any CPU |
| ✅ Ollama, LM Studio, API keys | ✅ No leaks between splits | ✅ LLM decisions as a tree | ✅ HTTP server and Docker |
| ✅ Hugging Face datasets | ✅ Overfitting check | ✅ Tokens per task and model | ✅ Publish to Hugging Face |
| ✅ LLM data, judge-checked | ✅ Health score | ✅ Test sets, Excel export | ✅ Teams that act as you type |
| ✅ Experiments and autopilot | ✅ Hugging Face rivals | ✅ Paged tables, live counts | ✅ SQLite or Postgres |
| ✅ Auto until ready for production | ✅ Ablations with matched seeds | ✅ Machine, GPU and memory live | ✅ Each model in its own container |
| ✅ Labels derived from content | ✅ Cost and break-even against LLMs | ✅ Waits for LLM limits, or goes local | ✅ Teams deploy like models |
| ✅ Ideas researched on the web | ✅ Next step for each failing check | ✅ GPU queue that cannot freeze the machine | ✅ Datasets published with answers |
| ✅ Open data found by asking | ✅ Leakage check for register snapshots | ✅ Sources the LLM discovers, saved | ✅ Tables cached once per machine |

## Command line

Installed as in [Run it](../README.md#run-it):

```bash
byosm build vad && byosm board vad        # no LLM needed: spec.yaml lists every data source
byosm export vad && byosm predict vad clip.wav
```

| command | what it does |
|---|---|
| `byosm "<task>" [--budget 6] [--publish repo]` | the whole thing: spec, benchmark, rivals, experiments, distill, export |
| `byosm auto <name>` | resume a specialist where it stopped |
| `byosm request-data <name>` | a template to fill (csv and xlsx) and the rows needed per label; written for you when no dataset fits |
| `byosm check-data <name> <file>` | check a filled table (format, coverage, leaks, likely mislabels), then train on it |
| `byosm new <name> "<task>"` | just compile the spec |
| `byosm build <name> [--exp vN] [--overfit]` | data, pretrain, train, evaluate, promote |
| `byosm eval <name>` · `byosm board <name>` | score everything, show the leaderboard |
| `byosm overfit <name> [--exp vN]` | is it overfitting? train vs test gap, calib vs test, train/test overlap, selection pressure, learning curve |
| `byosm improve <name> [--apply ids --run]` | diagnose, propose, run the next experiment |
| `byosm exp create <name> <exp> --set path=value` · `byosm promote <name> <exp>` | one-line ablations; choose what ships |
| `byosm suite add` · `byosm baseline add` | add a benchmark or a rival |
| `byosm distill <name> [--run]` | shrink a winner into a smaller student |
| `byosm export` · `byosm predict` · `byosm serve` | ONNX bundle, run it, local HTTP API |
| `byosm app [--token t --allow-host h]` | local web UI and API: specialists, live jobs, keys (`docker/compose.yaml` to self-host) |
| `byosm containerize <name> [--tag t]` | a small non-root Docker image that serves one exported bundle |
| `byosm publish <name> --repo <id>` | push to Hugging Face with a generated model card |
| `byosm bench <a> <b> ...` | specialists vs one generalist (not run yet, see [Goal and bet](assumptions.md)) |
| `byosm opendata find "<request>"` | the LLM searches the web for tables, adds the portals it discovers to your sources and searches them (CKAN and Socrata) |
| `byosm report [--check]` | regenerate the tables in [Results](results.md) |

Self-host with Docker: `BYOSM_APP_TOKEN=$(openssl rand -base64 32) docker compose -f docker/compose.yaml up`, then open
`http://<host>:8765/#token=<that token>`. The container runs read-only, without capabilities, as a non-root user; the
API refuses requests without the token. Checked on 2026-10-23: the image builds, the API answers 401 without the token,
and jobs run inside the read-only container.

## Modalities

`text`, `audio` (streaming, 10 ms frames), `image`, `document` (PDF or scan: text layer or OCR plus
the page image). Tasks: `classify` and `frame`.

## The bundle

`byosm export` writes `model.onnx`, `manifest.json` and `byosm_rt.py`. The runtime needs only `numpy`
and `onnxruntime` and gives every modality the same `predict`, `encode`, `stream().step` and `serve`.

## The LLM

It writes specs, writes and judges text data, analyses errors, and is the `llm` rival. Each role is set in
`byosm.yaml` and falls back to `default`. Here that's local Qwen 3.8 through Ollama:

```yaml
llm:
  default: {provider: openai, model: qwen3.8:27b, base_url: http://localhost:11434/v1, api_key: ollama, reasoning_effort: none}
```

`provider: openai` means any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenAI, OpenRouter).

## Extending

A modality is a class in `byosm/modalities/` with `@register`. A rival is a `Predictor` with
`@register_baseline`, or any `hf:<repo>` classifier in `baselines:`. An improve rule is a function with `@rule`. An LLM
provider is `register_provider`.

## Layout

```
byosm/
  cli/          commands
  spec/         spec, compiler, backbone catalog
  data/         Hugging Face sources, audio, synthetic text, store
  modalities/   text, audio, image, document
  train/        training loop, evaluation, teacher
  evals/        suites, scoring, board, selection
  improve/      diagnosis, rules, LLM analyst, costs
  auto/         autopilot
  predictors/   our models and rivals behind one interface
  export/       ONNX, int8, bundle, parity checks
  hub/          Hugging Face publishing, model cards
  llm/          provider-agnostic JSON calls
  bench/        specialists vs generalist
  report/       result tables in docs/results.md
  db/           SQLAlchemy schema: SQLite by default, Postgres via DATABASE_URL
specialists/<name>/   local weights and exports only; everything else is in the database
docs/design-notes.md  why things are the way they are
```

Datasets, weights and exports are never committed.

## Licence

CC BY-NC 4.0 ([LICENSE](../LICENSE)), for the code and the published models. The VAD's training data
includes ESC-50 (CC BY-NC): check data licences before any other use.

<p align="center"><a href="../README.md">Tour</a> · <a href="assumptions.md">Goal and bet</a> · <a href="methodology.md">Methodology</a> · <a href="results.md">Results</a> · <b>Details</b></p>
