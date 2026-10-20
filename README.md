# BYOSM: Build Your Own Specialist Model

## Goal

One sentence and one command give you a small specialist model that beats general models, LLMs included, in its
domain, measured on a public benchmark and small enough to run on any CPU.

```bash
byosm auto doctype "classify business documents (PDF or scan) into invoice, letter, memo, form, ..." --budget 6
```

## Assumptions

The thesis, stated as hypotheses:

| | hypothesis | status |
|---|---|---|
| 1 | On a narrow task, a small model trained only for that task beats a general model, LLMs included. | partly tested against outside baselines (summary table below); **not yet compared with an LLM** |
| 2 | Selection can be honest: decide on calib, report test once, with a CI and a paired bootstrap. | built into promotion, the autopilot and the export check; **broken once by hand**: the training-recipe default was picked on test (see below) |
| 3 | Specialists are cheap: small enough to run side by side on a laptop CPU, with no GPU and no PyTorch. | size and CPU time per model below (timed in PyTorch) |

Not tested yet:

- **No specialist has been compared with an LLM.** The zero-shot comparison against local Qwen 3.8 is about to run.
- **The specialists-vs-one-generalist experiment (`byosm bench`) has not been run.**
- **The autopilot (`byosm auto`) has only run end to end against a stub LLM**, in the tests. The results below come
  from experiments run step by step.

## Results

The tables are generated from `specialists/*/board.json` by `byosm report`; the notes under them are written by
hand and carry no numbers. `byosm report --check` fails when a table is out of date or a block is unknown or
repeated, and the test suite runs it against this file.

<!-- report:summary -->
| task | benchmark (metric) | our model | our test score [95% CI] | best outside baseline (on calib) | its test score | ours minus baseline on test (paired bootstrap) | calib, the selection split: ours vs baseline |
|---|---|---|---|---|---|---|---|
| 16-way document type classification (RVL-CDIP classes) | chainyo/rvl-cdip (accuracy) | v2-ditft (109.55M) | 0.948 [0.941, 0.956] | hf:microsoft/dit-base-finetuned-rvlcdip (85.82M) | 0.940 [0.932, 0.948] | +0.008, P(better) = 1.00 | 0.935 vs 0.936 (behind) |
| 77-way banking intent classification | mteb/banking77 (accuracy) | final-s1 (22.74M) | 0.937 [0.928, 0.948] | no outside baseline scored yet | - | - | - |
| Per-10ms-frame speech vs non-speech on 16kHz mic audio | nccratliri/vad-human-ava-speech (f1) | v3 (0.10M) | 0.882 [0.863, 0.899] | silero (0.55M) | 0.865 [0.839, 0.887] | +0.016, P(better) = 1.00 | 0.814 vs 0.772 (ahead) |
<!-- /report:summary -->

### Voice activity detection ([bsye/byosm-vad](https://huggingface.co/bsye/byosm-vad))

<!-- report:vad -->
| model | built from | train rows | params | CPU | test f1 [95% CI] | calib f1 |
|---|---|---|---|---|---|---|
| **v3** (promoted) | v2 + `train.ema=true train.holdout_test=auto`; spec also differs in 19 other fields (data, document, train) | 8400 | 0.10M | 0.169/30ms | 0.882 [0.863, 0.899] | 0.814 |
| v2 | v1 + real-data-ava, utterance-labels; spec also differs in 1 other field (labels) | 8400 | 0.10M | 0.176/30ms | 0.876 [0.855, 0.895] | 0.803 |
| silero | outside baseline | - | 0.55M | 0.066/32ms | 0.865 [0.839, 0.887] | 0.772 |
| v1 | no recorded parent | 3000 | 0.10M | 0.170/30ms | 0.848 [0.818, 0.873] | 0.811 |
| v0 | no recorded parent | 3000 | 0.10M | 0.169/30ms | 0.845 [0.815, 0.868] | 0.802 |
| energy | outside baseline | - | - | - | 0.731 [0.674, 0.776] | 0.733 |

Suite `ava` (nccratliri/vad-human-ava-speech): 20 test and 8 calib recordings. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison.
<!-- /report:vad -->

- In-domain: v2 and v3 train on the train partition of AVA-Speech (film and TV audio, hash-disjoint from the test
  files). Silero has never seen that domain, so this is a win where the thesis says it should be, not everywhere.
- v2 came from `byosm improve` reading the board: train on real film audio, and label whole utterances.
- v3's recorded change is EMA plus `holdout_test: auto`, but the fold did not run (its primary suite was `builtin`,
  see train rows), and its spec differs from v2's in many more fields (built from): the gain over v2 is not
  attributable to one edit.
- The AVA suite has few recordings: read the paired bootstrap column, not CI overlap.

### Documents ([bsye/byosm-doctype](https://huggingface.co/bsye/byosm-doctype))

<!-- report:doctype -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **v2-ditft** (promoted) | v1 + `document.image_backbone=hf:microsoft/dit-base-finetuned-rvlcdip document.image_pool=pooler` | 34935 (+5823 builtin test) | 109.55M | 43.27ms | 0.948 [0.941, 0.956] | 0.935 |
| hf:microsoft/dit-base-finetuned-rvlcdip | outside baseline | - | 85.82M | 36.76ms | 0.940 [0.932, 0.948] | 0.936 |
| v1 | r2 + more-data | 34935 (+5823 builtin test) | 109.55M | 44.65ms | 0.918 [0.908, 0.928] | 0.930 |
| r2 | fused-dit + `train.holdout_test=auto`; spec also differs in 8 other fields (train) | 11520 (+1920 builtin test) | 109.55M | 48.11ms | 0.905 [0.894, 0.915] | 0.906 |
| fused-dit | fused + more-data, domain-encoder, train-longer | 9600 | 109.55M | 46.90ms | 0.884 [0.873, 0.895] | 0.904 |
| distill-r2 | r2 + distill; spec also differs in 4 other fields (document, train) | 11520 (+1920 builtin test) | 17.58M | 10.68ms | 0.882 [0.870, 0.893] | 0.894 |
| fused | spec.yaml + `document.route=fused`; spec also differs in 1 other field (evals) | 2400 | 45.70M | 24.39ms | 0.801 [0.787, 0.815] | 0.804 |
| text | spec.yaml + `document.route=text`; spec also differs in 1 other field (evals) | 2400 | 23.41M | 6.74ms | 0.741 [0.725, 0.756] | 0.728 |
| visual | spec.yaml + `document.route=visual`; spec also differs in 1 other field (evals) | 2400 | 22.75M | 18.83ms | 0.645 [0.628, 0.662] | 0.654 |

Suite `rvl` (chainyo/rvl-cdip): 3200 test and 800 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:doctype -->

- v2-ditft starts its visual branch from DiT-base's own fine-tuned checkpoint and adds a text (OCR) branch. The claim is
  "DiT plus text beats DiT", not "beat DiT from less data": the checkpoint brings DiT's labelled training set.
- On calib, the selection split, DiT-base leads (summary table), so by the selection rule this is not a win yet.
- RVL-CDIP has known label noise, which caps every model's accuracy on it.
- distill-r2 is a tiny-BERT + DeiT-tiny student of r2; `text` and `visual` are the single-modality ablations of
  `fused`.

### Banking intent ([bsye/byosm-intent](https://huggingface.co/bsye/byosm-intent))

<!-- report:intent -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **final-s1** (promoted) | ampemallrd-s1 + `train.holdout_test=auto` | 8964 (+1470 builtin test) | 22.74M | 2.36ms | 0.937 [0.928, 0.948] | 0.948 |
| ampemallrd-s1† | spec.yaml + `train.layer_decay_floor=0.5 train.probe_epochs=0 train.ema=true train.seed=1`; spec also differs in 3 other fields (baselines, document, train) | 7494 | 22.74M | 3.37ms | 0.934 [0.924, 0.943] | 0.943 |

Suite `b77` (mteb/banking77): 2460 test and 616 calib items. Selection uses calib; test is only reported. Every scored experiment is listed, except 27 seed replicates off the promoted lineage, which are summarised in the recipe table (`intent:seeds`). CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on. † = scored on a different version of this suite or its decision policy than the promoted model; rerun `byosm eval` to refresh.
<!-- /report:intent -->

- No outside baseline has been scored by the same code yet, so there is no comparison here; the training-recipe
  ablation below is measured on this benchmark.

## Methodology and benchmarks

```
"one sentence" ─▶ spec.yaml ─▶ data ─▶ LeJEPA pretrain ─▶ train ─▶ score every suite ─▶ promote on calib ─▶ export
                     ▲                                                                       │
                     └──────────── byosm improve: one spec edit, with its evidence ◀─────────┘
```

1. **Define.** An LLM compiles the sentence into `spec.yaml`: labels with definitions, objective, data plan, backbone,
   decision policy, eval suites, baselines. You read and edit it.
2. **Data.** Hugging Face datasets mapped onto your labels, real segment-labelled audio, procedural audio mixtures,
   LLM-written text (double-labelled by a blind judge), or local files. Cached by a hash of the data plan.
3. **Pretrain.** LeJEPA self-supervised adaptation on unlabelled in-domain inputs (multi-view prediction + SIGReg).
4. **Train.** A small head on the backbone, checkpoint chosen on dev.
5. **Evaluate.** Each eval suite is frozen and split into calib and test by a stable hash of file names. Experiments
   and baselines (`silero`, `energy`, `llm`, any `hf:<repo>` classifier) go through the same scorer, and the decision
   policy (threshold, or streaming hysteresis) is calibrated on calib.
6. **Improve.** `byosm improve` diagnoses the board and proposes spec edits, each with the evidence behind it.

Selection rules:

- Promotion, the autopilot's keep/revert and stopping rule, and the int8 export check all decide on **calib**. The calib
  score is cross-fitted, so a model is not rewarded for a threshold tuned on the items it is scored on.
- **Test is only reported**, with a 95% bootstrap CI. Comparisons are a **paired bootstrap** over the same files or
  items ("P(better)"), not a difference of two point estimates.
- The autopilot changes one thing per round, keeps it only if it wins on calib, and counts it as progress only when
  the paired bootstrap clears `SIGNIFICANT_P_BETTER` (`byosm/auto/loop.py`).

### The training recipe, measured

Banking77 with MiniLM-L6. Only `train` settings change, but `holdout_test: auto` also adds the builtin suite's
held-out split to training (train rows column), so those recipes see more data.

<!-- report:intent:seeds -->
| recipe | seeds | train rows | ema | holdout_test | layer_decay_floor | precision | probe_epochs | test accuracy % | calib accuracy % |
|---|---|---|---|---|---|---|---|---|---|
| full† | 3 | 7494 | true | keep | 0.5 | auto | 1 | 92.97 ± 0.29 | 94.70 ± 0.19 |
| **final**† | 3 | 8964 (+1470 builtin test) | true | auto | 0.5 | auto | 0 | 93.44 ± 0.27 | 94.37 ± 0.37 |
| no-ema† | 2 | 7494 | false | - | 0.1 | auto | 1 | 93.13 ± 0.75 | 94.24 ± 0.57 |
| no-probe† | 2 | 7494 | true | - | 0.1 | auto | 0 | 93.25 ± 0.34 | 94.24 ± 0.34 |
| ampemallrd† | 3 | 7494 | true | keep | 0.5 | auto | 0 | 93.14 ± 0.21 | 94.10 ± 0.37 |
| ampllrd† | 3 | 7494 | false | keep | 0.5 | auto | 0 | 92.59 ± 0.42 | 93.94 ± 0.52 |
| no-llrd† | 2 | 7494 | true | - | 1.0 | auto | 1 | 92.99 ± 0.14 | 93.91 ± 0.11 |
| new† | 2 | 7494 | true | - | 0.1 | auto | 1 | 92.72 ± 0.23 | 93.83 ± 0.69 |
| ampema† | 3 | 7494 | true | keep | 1.0 | auto | 0 | 93.01 ± 0.21 | 93.72 ± 1.06 |
| amp† | 3 | 7494 | false | keep | 1.0 | auto | 0 | 92.53 ± 0.31 | 93.67 ± 0.56 |
| old† | 3 | 7494 | false | - | 1.0 | fp32 | 0 | 92.52 ± 0.25 | 93.67 ± 0.56 |

Experiments named `<recipe>-s<seed>` on suite `b77`, mean ± sample sd over seeds, sorted by calib (the selection split); top on calib: `full`. Columns after train rows are the `train` settings that differ between recipes; - = not set (the code default when it ran). +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on. † = scored on a different version of this suite or its decision policy than the promoted model; rerun `byosm eval` to refresh.
<!-- /report:intent:seeds -->

- The code default (`TrainCfg`: EMA, layer decay floor 0.5, no probe epoch, mixed precision, `holdout_test: auto`)
  was chosen by hand **on test means**, which breaks hypothesis 2; the note above names the recipe that leads on calib.
- Calib differences between recipes are of the order of one sample sd, and with so few seeds the sd itself is rough,
  so the best recipe is not resolved; re-choosing the default on calib, with more seeds, is open.
- `final` (bold) is the promoted lineage. Rows marked † were scored on an older suite version than the promoted model.

## Quickstart

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
byosm build vad && byosm board vad        # reproducible without an LLM: spec.yaml lists every data source
byosm export vad && byosm predict vad clip.wav
```

| command | what it does |
|---|---|
| `byosm auto <name> "<task>" [--budget 6] [--margin 0] [--publish repo]` | spec, benchmark and baselines, experiments, distill, export |
| `byosm init` · `byosm new <name> "<task>"` | create `byosm.yaml`; compile a spec from a sentence |
| `byosm suite add <name> <suite> --hf <id> [--format segment_files] [--primary]` | add a frozen eval suite |
| `byosm baseline add <name> <baseline>` | add a registered competitor (`byosm baseline available`) |
| `byosm build <name> [--exp vN]` | data, pretrain, train, evaluate, promote |
| `byosm eval <name>` · `byosm board <name> [--detail]` | score and compare |
| `byosm improve <name> [--apply ids --run]` | diagnose, propose, run the next experiment |
| `byosm exp create <name> <exp> --set path=value` · `byosm promote <name> <exp>` | one-line ablations; choose what ships |
| `byosm distill <name> [--teacher exp] [--run]` | shrink a teacher into a small student |
| `byosm export <name>` · `byosm predict <name> <input>` · `byosm serve <names...>` | ONNX bundle, run it, local HTTP API |
| `byosm publish <name> --repo <id> [--public] [--dry-run]` | Hugging Face bundle with a generated model card |
| `byosm bench <a> <b> ...` | specialists vs one shared-backbone generalist (not run yet) |
| `byosm report [--check]` | regenerate the result tables in this README |

## Details

**Modalities.** `text`, `audio` (streaming, per-10 ms frames), `image` and `document` (PDF or scan; text layer or OCR,
plus the rendered page, fused). Task types: `classify` and `frame`.

**Exported bundle.** `byosm export` writes `model.onnx` (+ `stream.onnx` for streaming audio), `manifest.json` and
`byosm_rt.py`, which needs only `numpy` and `onnxruntime` (plus `tokenizers` for text, Pillow for images) and offers
the same `predict`, `encode`, `stream().step` and `serve` for every modality.

**LLM configuration.** An LLM compiles specs, writes and judges text data, analyses errors, and is the `llm` baseline.
Each role (`spec`, `generate`, `judge`, `eval`, `analyze`, `baseline`) is set in `byosm.yaml` and falls back to
`default`. This repository's default is local Qwen 3.8 through Ollama:

```yaml
llm:
  default: {provider: openai, model: qwen3.8:27b, base_url: http://localhost:11434/v1, api_key: ollama, reasoning_effort: none}
```

`provider: openai` is any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenAI, OpenRouter);
`byosm.llm.register_provider(name, factory)` adds other providers.

**Extending.**

- Modality: a class in `byosm/modalities/` registered with `@register`.
- Baseline: a `Predictor` registered with `@register_baseline` in `byosm/predictors/`, or any `hf:<repo>` classifier
  listed under `baselines:` in `spec.yaml`.
- Improvement rule: a function decorated with `@rule` in `byosm/improve/`.
- LLM provider: `register_provider`.

**Layout.**

```
byosm/
  cli/          Typer commands (define, train, iterate, ship, report)
  spec/         TaskSpec, spec compiler, backbone catalog
  data/         Hugging Face sources, audio mixing, segments, synthetic text, store
  modalities/   text, audio, image, document/
  train/        supervised training loop, evaluation, teacher
  evals/        suites, scoring and calibration, board, selection
  improve/      diagnosis, rules, LLM analyst, cost estimates
  auto/         autopilot loop and its report
  predictors/   experiments and baselines behind one interface
  export/       ONNX graph, int8 quantisation, bundle, parity checks
  hub/          Hugging Face publishing and model card
  llm/          provider-agnostic structured-JSON calls
  bench/        specialists vs generalist, latency
  report/       README result tables from the boards (byosm report)
  config.py  discover.py  distill.py  jepa.py  metrics.py  model.py  pipeline.py  rt.py  runtime.py  text.py
specialists/<name>/  spec.yaml, experiments/*/, baselines/, board.json, promoted
benchmarks/   vad_vs_silero.py, an older standalone VAD vs Silero script, not used for the results above
bench/        its saved output
```

Datasets, weights and exports are not committed.

**Licence.** CC BY-NC 4.0 (see [LICENSE](LICENSE)), for code and published models. The VAD's training data includes
ESC-50 (CC BY-NC); check data licences before any other use.
