# BYOSM: Build Your Own Specialist Model

```
 ____  __   __  ___   ____   __  __
| __ ) \ \ / / / _ \ / ___| |  \/  |
|  _ \  \ V / | | | |\___ \ | |\/| |
| |_) |  | |  | |_| | ___) || |  | |
|____/   |_|   \___/ |____/ |_|  |_|

  one sentence  ->  a small model that beats the big ones at its job
```

**Models on Hugging Face:** [bsye](https://huggingface.co/bsye) ·
[byosm-doctype](https://huggingface.co/bsye/byosm-doctype) ·
[byosm-intent](https://huggingface.co/bsye/byosm-intent) ·
[byosm-vad](https://huggingface.co/bsye/byosm-vad)

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
| 1 | On a narrow task, a small model trained only for that task beats a general model, LLMs included. | **holds on 5 public benchmarks**: ahead of the best outside model on test in all 5 (on calib, DiT-base still leads documents by a hair), and far ahead of Qwen 3.8 27B zero-shot on the 4 tasks an LLM can do (summary table and each task's table) |
| 2 | Selection can be honest: decide on calib, report test once, with a CI and a paired bootstrap. | built into promotion, the autopilot and the export check; **broken once by hand**: the training-recipe default was picked on test (see below) |
| 3 | Specialists are cheap: small enough to run side by side on a laptop CPU, with no GPU and no PyTorch. | size and CPU time per model below (timed in PyTorch) |

Not tested yet:

- **Specialists vs one shared generalist (`byosm bench`) has not been run.** It scores on each specialist's builtin
  test split, which `holdout_test: auto` folds into the promoted models' training, so it must move to the public suites
  before its numbers mean anything.
- **The autopilot (`byosm auto`) has run end to end twice**, from one sentence each, with Qwen 3.8 as its LLM: emotion
  won inside its budget; topic ran one experiment, did not win, and won after two more-data steps run by hand.
  VAD, documents and intent were built step by step.

## Results

The tables are generated from `specialists/*/board.json` by `byosm report`; the notes under them are written by
hand and carry no numbers. `byosm report --check` fails when a table is out of date or a block is unknown or
repeated, and the test suite runs it against this file.

<!-- report:summary -->
| task | benchmark (metric) | our model | our test score [95% CI] | best outside baseline (on calib) | its test score | ours minus baseline on test (paired bootstrap) | calib, the selection split: ours vs baseline |
|---|---|---|---|---|---|---|---|
| 16-way document type classification (RVL-CDIP classes) | chainyo/rvl-cdip (accuracy) | v2-ditft (109.55M) | 0.948 [0.941, 0.956] | hf:microsoft/dit-base-finetuned-rvlcdip (85.82M) | 0.940 [0.932, 0.948] | +0.008, P(better) = 1.00 | 0.935 vs 0.936 (behind) |
| Classify the primary emotion expressed in a short English text (single sentence, tweet-style or short social media string) into one of six named emotions (sadness, joy, love, anger, fear, surprise). Any text that does not clearly express one of these dominates falls into a catch-all neutral-other class (including neutral statements, irony/sarcasm without a dominant named emotion, ambiguity, or non-English). | dair-ai/emotion (accuracy) | v1 (22.72M) | 0.898 [0.883, 0.913] | hf:j-hartmann/emotion-english-distilroberta-base (82.12M) | 0.838 [0.819, 0.856] | +0.060, P(better) = 1.00 | 0.904 vs 0.813 (ahead) |
| 77-way banking intent classification | mteb/banking77 (accuracy) | final-s1 (22.74M) | 0.937 [0.928, 0.948] | llm (-) | 0.772 [0.756, 0.788] | +0.165, P(better) = 1.00 | 0.948 vs 0.713 (ahead) |
| Classify an English news-headline+summary pair into one of four topics: world, sports, business, science/technology, with an explicit 'other' catch-all for items that fit none of the four. | fancyzhx/ag_news (accuracy) | v3 (22.72M) | 0.947 [0.937, 0.955] | hf:mrm8488/bert-mini-finetuned-age_news-classification (11.17M) | 0.930 [0.919, 0.939] | +0.017, P(better) = 1.00 | 0.955 vs 0.934 (ahead) |
| Per-10ms-frame speech vs non-speech on 16kHz mic audio | nccratliri/vad-human-ava-speech (f1) | v3 (0.10M) | 0.882 [0.863, 0.899] | silero (0.55M) | 0.865 [0.839, 0.887] | +0.016, P(better) = 1.00 | 0.814 vs 0.772 (ahead) |
<!-- /report:summary -->

### Voice activity detection ([bsye/byosm-vad](https://huggingface.co/bsye/byosm-vad))

<!-- report:vad -->
| model | built from | train rows | params | CPU | test f1 [95% CI] | calib f1 |
|---|---|---|---|---|---|---|
| **v3** (promoted) | v2 + `train.ema=true train.holdout_test=auto` | 8400 | 0.10M | 0.169/30ms | 0.882 [0.863, 0.899] | 0.814 |
| v2 | v1 + real-data-ava, utterance-labels | 8400 | 0.10M | 0.176/30ms | 0.876 [0.855, 0.895] | 0.803 |
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
| r2 | fused-dit + `train.holdout_test=auto` | 11520 (+1920 builtin test) | 109.55M | 48.11ms | 0.905 [0.894, 0.915] | 0.906 |
| fused-dit | fused + more-data, domain-encoder, train-longer | 9600 | 109.55M | 46.90ms | 0.884 [0.873, 0.895] | 0.904 |
| distill-r2 | r2 + distill | 11520 (+1920 builtin test) | 17.58M | 10.68ms | 0.882 [0.870, 0.893] | 0.894 |
| fused | spec.yaml + `document.route=fused` | 2400 | 45.70M | 24.39ms | 0.801 [0.787, 0.815] | 0.804 |
| llm | outside baseline | - | - | - | 0.744 [0.729, 0.759] | 0.744 |
| text | spec.yaml + `document.route=text` | 2400 | 23.41M | 6.74ms | 0.741 [0.725, 0.756] | 0.728 |
| visual | spec.yaml + `document.route=visual` | 2400 | 22.75M | 18.83ms | 0.645 [0.628, 0.662] | 0.654 |

Suite `rvl` (chainyo/rvl-cdip): 3200 test and 800 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:doctype -->

- v2-ditft starts its visual branch from DiT-base's own fine-tuned checkpoint and adds a text (OCR) branch. The claim is
  "DiT plus text beats DiT", not "beat DiT from less data": the checkpoint brings DiT's labelled training set.
- On calib, the selection split, DiT-base leads (summary table), so by the selection rule this is not a win yet.
- RVL-CDIP has known label noise, which caps every model's accuracy on it.
- `llm` is Qwen 3.8 27B, zero-shot, shown the page images and their OCR text.
- distill-r2 is a tiny-BERT + DeiT-tiny student of r2; `text` and `visual` are the single-modality ablations of
  `fused`.

### Banking intent ([bsye/byosm-intent](https://huggingface.co/bsye/byosm-intent))

<!-- report:intent -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **final-s1** (promoted) | ampemallrd-s1 + `train.holdout_test=auto` | 8964 (+1470 builtin test) | 22.74M | 2.36ms | 0.937 [0.928, 0.948] | 0.948 |
| ampemallrd-s1† | spec.yaml + `train.layer_decay_floor=0.5 train.probe_epochs=0 train.ema=true train.seed=1` | 7494 | 22.74M | 3.37ms | 0.934 [0.924, 0.943] | 0.943 |
| llm | outside baseline | - | - | - | 0.772 [0.756, 0.788] | 0.713 |

Suite `b77` (mteb/banking77): 2460 test and 616 calib items. Selection uses calib; test is only reported. Every scored experiment is listed, except 27 seed replicates off the promoted lineage, which are summarised in the recipe table (`intent:seeds`). CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on. † = scored on a different version of this suite or its decision policy than the promoted model; rerun `byosm eval` to refresh.
<!-- /report:intent -->

- The outside baseline is Qwen 3.8 27B, zero-shot, given the 77 label definitions (thinking off). No fine-tuned
  Banking77 classifier has been scored yet.
- The training-recipe ablation below is measured on this benchmark.

### Emotion (autopilot, one sentence)

<!-- report:emotion -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **v1** (promoted) | no recorded parent | 5241 (+781 builtin test) | 22.72M | 3.61ms | 0.898 [0.883, 0.913] | 0.904 |
| hf:j-hartmann/emotion-english-distilroberta-base | outside baseline | - | 82.12M | 25.05ms | 0.838 [0.819, 0.856] | 0.813 |
| llm | outside baseline | - | - | - | 0.549 [0.519, 0.572] | 0.489 |
| hf:SamLowe/roberta-base-go_emotions | outside baseline | - | 124.67M | 26.37ms | 0.548 [0.523, 0.570] | 0.519 |

Suite `bench` (dair-ai/emotion): 1492 test and 374 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:emotion -->

- Built by `byosm auto` from one sentence: it chose the benchmark and the baselines, trained on dair-ai/emotion's own
  train split plus text written by Qwen, and exported, inside its budget.
- The spec adds a catch-all `other` that the benchmark never uses; every model, the LLM included, is scored inside the
  benchmark's own labels ([design notes](docs/design-notes.md#evaluation-and-selection)).
- j-hartmann's model has no `love` class, and SamLowe's was trained on Reddit (GoEmotions), not tweets.

### News topic (autopilot, then two data steps)

<!-- report:topic -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **v3** (promoted) | v2 + `data.hf_sources.3.per_label=30000` | 107112 (+17112 builtin test) | 22.72M | 3.24ms | 0.947 [0.937, 0.955] | 0.955 |
| hf:mrm8488/bert-mini-finetuned-age_news-classification | outside baseline | - | 11.17M | 1.47ms | 0.930 [0.919, 0.939] | 0.934 |
| v2 | v1 + more-data | 14383 (+2383 builtin test) | 22.72M | 3.10ms | 0.921 [0.910, 0.930] | 0.925 |
| hf:Maddyhf1/ag-news-topic-classification-distilbert | outside baseline | - | 66.96M | 9.07ms | 0.914 [0.902, 0.923] | 0.917 |
| v1 | no recorded parent | 5999 (+917 builtin test) | 22.72M | 3.19ms | 0.903 [0.891, 0.913] | 0.914 |
| llm | outside baseline | - | - | - | 0.814 [0.799, 0.830] | 0.800 |

Suite `bench` (fancyzhx/ag_news): 2560 test and 640 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:topic -->

- v1 is the autopilot's: a sample of AG News plus text written by Qwen; its budget ended after one experiment.
- v2 (`byosm improve`: more-data) and v3 (every AG News training row, by hand) each won on calib before
  promotion. Both outside baselines were fine-tuned on the full AG News train split.

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
| no-ema† | 2 | 7494 | false | - | 0.1 | auto | 1 | 93.13 ± 0.75 | 94.24 ± 0.57 |
| no-probe† | 2 | 7494 | true | - | 0.1 | auto | 0 | 93.25 ± 0.34 | 94.24 ± 0.34 |
| final† | 2 | 8964 (+1470 builtin test) | true | auto | 0.5 | auto | 0 | 93.29 ± 0.11 | 94.16 ± 0.00 |
| ampemallrd† | 3 | 7494 | true | keep | 0.5 | auto | 0 | 93.14 ± 0.21 | 94.10 ± 0.37 |
| ampllrd† | 3 | 7494 | false | keep | 0.5 | auto | 0 | 92.59 ± 0.42 | 93.94 ± 0.52 |
| no-llrd† | 2 | 7494 | true | - | 1.0 | auto | 1 | 92.99 ± 0.14 | 93.91 ± 0.11 |
| new† | 2 | 7494 | true | - | 0.1 | auto | 1 | 92.72 ± 0.23 | 93.83 ± 0.69 |
| ampema† | 3 | 7494 | true | keep | 1.0 | auto | 0 | 93.01 ± 0.21 | 93.72 ± 1.06 |
| amp† | 3 | 7494 | false | keep | 1.0 | auto | 0 | 92.53 ± 0.31 | 93.67 ± 0.56 |
| old† | 3 | 7494 | false | - | 1.0 | fp32 | 0 | 92.52 ± 0.25 | 93.67 ± 0.56 |

Experiments named `<recipe>-s<seed>` on suite `b77`, mean ± sample sd over seeds, sorted by calib (the selection split); only seeds scored on the same suite version are averaged (a recipe scored on several gets one row per version, `@<fingerprint>`; a seed alone on its version stays in the main table; † rows are listed last and never ranked against the promoted version); no recipe has seeds on the promoted model's suite version. Columns after train rows are the `train` settings that differ between recipes; - = not set (the code default when it ran). +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on. † = scored on a different version of this suite or its decision policy than the promoted model; rerun `byosm eval` to refresh.
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
| `byosm bench <a> <b> ...` | specialists vs one shared-backbone generalist (not run yet, see Assumptions) |
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
```

Datasets, weights and exports are not committed.

**Licence.** CC BY-NC 4.0 (see [LICENSE](LICENSE)), for code and published models. The VAD's training data includes
ESC-50 (CC BY-NC); check data licences before any other use.
