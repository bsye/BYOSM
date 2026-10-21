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

Describe a task in one sentence. Get back a small model that does that one job better than general models, LLMs
included, proven on a public benchmark, and fast enough to run on any CPU.

```bash
byosm "classify the emotion of a short English text: joy, love, sadness, anger, fear or surprise"
```

That one command names the task, works out what kind of model it needs (text, audio, image or document; one label
per input or one per word or frame), writes the spec, finds the benchmark and the models to beat, trains, keeps going until it wins or
runs out of time, and exports an ONNX bundle.

## The bet

| | we bet that | where it stands |
|---|---|---|
| 1 | A small model trained for one job beats a general model at that job, LLMs included. | **Holds on every public benchmark so far.** Ahead of the best outside model on test everywhere (on names, only just), and far ahead of Qwen 3.8 27B on every task an LLM can do. On documents, DiT-base still edges us on calib. |
| 2 | We can stay honest: choose on calib, report test once, compare with a paired bootstrap. | **Built in** to promotion, the autopilot and export. Broken once by hand: the training-recipe default was picked on test (below). |
| 3 | Specialists are cheap: they run side by side on a laptop CPU, no GPU, no PyTorch. | **Yes.** Size and CPU time are in every table. |

Still open:

- **Specialists vs one shared generalist** (`byosm bench`). Not run yet: it grades on a split the promoted models
  trained on, so it has to move to the public benchmarks first.
- **The autopilot on its own.** Two full runs from one sentence: emotion won inside its budget; topic needed two more
  data steps by hand. VAD, documents and intent were built step by step.

## Results

Every table below is generated from `specialists/*/board.json` by `byosm report`, never typed by hand.
`byosm report --check` fails the tests if one is stale.

<!-- report:summary -->
| task | benchmark (metric) | our model | our test score [95% CI] | best outside baseline (on calib) | its test score | ours minus baseline on test (paired bootstrap) | calib, the selection split: ours vs baseline |
|---|---|---|---|---|---|---|---|
| 16-way document type classification (RVL-CDIP classes) | chainyo/rvl-cdip (accuracy) | v2-ditft (109.55M) | 0.948 [0.941, 0.956] | hf:microsoft/dit-base-finetuned-rvlcdip (85.82M) | 0.940 [0.932, 0.948] | +0.008, P(better) = 1.00 | 0.935 vs 0.936 (behind) |
| Classify the primary emotion expressed in a short English text (single sentence, tweet-style or short social media string) into one of six named emotions (sadness, joy, love, anger, fear, surprise). Any text that does not clearly express one of these dominates falls into a catch-all neutral-other class (including neutral statements, irony/sarcasm without a dominant named emotion, ambiguity, or non-English). | dair-ai/emotion (accuracy) | v1 (22.72M) | 0.898 [0.883, 0.913] | hf:j-hartmann/emotion-english-distilroberta-base (82.12M) | 0.838 [0.819, 0.856] | +0.060, P(better) = 1.00 | 0.904 vs 0.813 (ahead) |
| 77-way banking intent classification | mteb/banking77 (accuracy) | final-s1 (22.74M) | 0.937 [0.928, 0.948] | llm (-) | 0.772 [0.756, 0.788] | +0.165, P(better) = 1.00 | 0.948 vs 0.713 (ahead) |
| Named entity recognition (CoNLL-2003 tags) | BramVanroy/conll2003 (span_f1) | v3 (70.69M) | 0.913 [0.903, 0.921] | hf:dslim/bert-base-NER (107.73M) | 0.910 [0.900, 0.918] | +0.003, P(better) = 0.86 | 0.928 vs 0.925 (ahead) |
| Classify an English news-headline+summary pair into one of four topics: world, sports, business, science/technology, with an explicit 'other' catch-all for items that fit none of the four. | fancyzhx/ag_news (accuracy) | v3 (22.72M) | 0.947 [0.937, 0.955] | hf:mrm8488/bert-mini-finetuned-age_news-classification (11.17M) | 0.930 [0.919, 0.939] | +0.017, P(better) = 1.00 | 0.955 vs 0.934 (ahead) |
| Per-10ms-frame speech vs non-speech on 16kHz mic audio | nccratliri/vad-human-ava-speech (f1) | v3 (0.10M) | 0.882 [0.863, 0.899] | silero (0.55M) | 0.865 [0.839, 0.887] | +0.016, P(better) = 1.00 | 0.814 vs 0.772 (ahead) |
<!-- /report:summary -->

### Voice activity ([bsye/byosm-vad](https://huggingface.co/bsye/byosm-vad))

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

- Trained on the train part of AVA-Speech (film and TV audio), kept apart from the test files. Silero never saw that
  domain: a win where the thesis says it should be.
- v2 came straight from `byosm improve`: use real film audio, label whole utterances.
- v3 changed more than its one recorded edit, so its gain over v2 can't be pinned on EMA alone.
- Few recordings in the suite: trust the paired bootstrap, not overlapping CIs.

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

- v2-ditft starts its visual side from DiT-base's fine-tuned checkpoint and adds the OCR text. The claim is
  "DiT plus text beats DiT", not "beat DiT with less data".
- On calib, DiT-base is still a hair ahead, so by our own rule this is not a clean win yet.
- `llm` is Qwen 3.8 27B, zero-shot, looking at the page images and their OCR text.
- RVL-CDIP has known label noise, which caps every model.

### Banking intent ([bsye/byosm-intent](https://huggingface.co/bsye/byosm-intent))

<!-- report:intent -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **final-s1** (promoted) | ampemallrd-s1 + `train.holdout_test=auto` | 8964 (+1470 builtin test) | 22.74M | 2.36ms | 0.937 [0.928, 0.948] | 0.948 |
| ampemallrd-s1† | spec.yaml + `train.layer_decay_floor=0.5 train.probe_epochs=0 train.ema=true train.seed=1` | 7494 | 22.74M | 3.37ms | 0.934 [0.924, 0.943] | 0.943 |
| llm | outside baseline | - | - | - | 0.772 [0.756, 0.788] | 0.713 |

Suite `b77` (mteb/banking77): 2460 test and 616 calib items. Selection uses calib; test is only reported. Every scored experiment is listed, except 27 seed replicates off the promoted lineage, which are summarised in the recipe table (`intent:seeds`). CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on. † = scored on a different version of this suite or its decision policy than the promoted model; rerun `byosm eval` to refresh.
<!-- /report:intent -->

- `llm` is Qwen 3.8 27B, zero-shot, given all 77 label definitions. No fine-tuned Banking77 model has been scored
  against us yet.

### Emotion: one sentence, end to end

<!-- report:emotion -->
| model | built from | train rows | params | CPU | test accuracy [95% CI] | calib accuracy |
|---|---|---|---|---|---|---|
| **v1** (promoted) | no recorded parent | 5241 (+781 builtin test) | 22.72M | 3.61ms | 0.898 [0.883, 0.913] | 0.904 |
| hf:j-hartmann/emotion-english-distilroberta-base | outside baseline | - | 82.12M | 25.05ms | 0.838 [0.819, 0.856] | 0.813 |
| llm | outside baseline | - | - | - | 0.549 [0.519, 0.572] | 0.489 |
| hf:SamLowe/roberta-base-go_emotions | outside baseline | - | 124.67M | 26.37ms | 0.548 [0.523, 0.570] | 0.519 |

Suite `bench` (dair-ai/emotion): 1492 test and 374 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:emotion -->

- Built entirely by `byosm auto`: it picked the benchmark and the rivals, trained on dair-ai/emotion's train split
  plus text written by Qwen, and exported.
- The spec has an extra `other` label the benchmark never uses, so every model is scored on the benchmark's own labels
  ([why](docs/design-notes.md#evaluation-and-selection)).
- j-hartmann's model has no `love` class; SamLowe's learned on Reddit, not tweets.

### News topic: autopilot, then more data

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

- v1 is the autopilot's, from a sample of AG News plus Qwen text. Its budget ran out after one experiment.
- v2 (`byosm improve`: more data) and v3 (every AG News training row) each won on calib before being promoted. Both
  rivals were fine-tuned on the full AG News train split, so this is a fair fight.

### Names in text: tagging every word

<!-- report:ner -->
| model | built from | train rows | params | CPU | test span_f1 [95% CI] | calib span_f1 |
|---|---|---|---|---|---|---|
| **v3** (promoted) | v2 + `backbone=hf:microsoft/deberta-v3-xsmall` | 15181 (+2214 builtin test) | 70.69M | 9.99ms | 0.913 [0.903, 0.921] | 0.928 |
| hf:dslim/bert-base-NER | outside baseline | - | 107.73M | 11.33ms | 0.910 [0.900, 0.918] | 0.925 |
| v2 | v1 + new-split, train-longer | 15181 (+2214 builtin test) | 65.20M | 5.37ms | 0.894 [0.883, 0.904] | 0.922 |
| v1 | no recorded parent | 12320 (+1790 builtin test) | 65.20M | 5.92ms | 0.890 [0.878, 0.900] | 0.907 |
| llm | outside baseline | - | - | - | 0.577 [0.560, 0.596] | 0.572 |

Suite `conll` (BramVanroy/conll2003): 2762 test and 691 calib items. Selection uses calib; test is only reported. Every scored experiment is listed. CPU is PyTorch time per input, or per streaming chunk, timed when the row was last scored, so rows scored in different sessions are not a controlled speed comparison. +N builtin test = the builtin suite's held-out split was folded into training (`train.holdout_test`); the benchmark's own test split is never trained on.
<!-- /report:ner -->

- Scored as whole names (CoNLL span F1), with each sentence as one bootstrap unit.
- v2 (`byosm improve`) added CoNLL's validation split, which no suite reads, and trained longer. v3 swapped the backbone
  for DeBERTa-v3-xsmall. dslim's model is BERT-base fine-tuned on the same CoNLL train split.
- This is the thinnest lead on the board: ahead on calib and test, but the paired bootstrap is not yet sure.
- Qwen tags word lists in batches; it finds about as many names as there are and gets many wrong.

## How it works

```
"one sentence" ─▶ spec.yaml ─▶ data ─▶ LeJEPA pretrain ─▶ train ─▶ score every suite ─▶ promote on calib ─▶ export
                     ▲                                                                       │
                     └──────────── byosm improve: one spec edit, with its evidence ◀─────────┘
```

| step | what happens |
|---|---|
| **Define** | An LLM turns the sentence into `spec.yaml`: labels with definitions, objective, data plan, backbone, decision policy, benchmarks, rivals. You can read and edit it. |
| **Data** | Hugging Face datasets mapped onto your labels, real labelled audio, audio mixtures, LLM-written text checked by a blind judge, or your own files. Cached by content. |
| **Pretrain** | LeJEPA self-supervised adaptation on unlabelled in-domain inputs. |
| **Train** | A small head on the backbone, best checkpoint chosen on dev. |
| **Evaluate** | Every model, ours or a rival (`silero`, `energy`, `llm`, any `hf:<repo>`), goes through the same scorer on frozen calib and test splits. |
| **Improve** | `byosm improve` reads the board and proposes the next spec edit, with the evidence behind it. |

How we keep it honest:

- **Choose on calib.** Promotion, the autopilot's keep/revert and the int8 export check all decide on calib, and the
  calib score is cross-fitted so a threshold never grades itself.
- **Report test once**, with a 95% bootstrap CI.
- **Compare paired.** Every "ours vs theirs" is a paired bootstrap over the same items (P(better)), not two point
  estimates side by side.

### The training recipe, measured

Banking77 with MiniLM-L6, only the `train` settings change. Recipes with `holdout_test: auto` also train on the
builtin suite's held-out split, so they see more data.

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

- The current default (EMA, layer decay floor 0.5, no probe epoch, mixed precision, `holdout_test: auto`) was picked
  on **test** means. That breaks bet 2; re-picking it on calib with more seeds is open.
- The differences between recipes are within about one seed's noise: no recipe is a clear winner yet.

## Quickstart

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
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
| `byosm build <name> [--exp vN]` | data, pretrain, train, evaluate, promote |
| `byosm eval <name>` · `byosm board <name>` | score everything, show the leaderboard |
| `byosm improve <name> [--apply ids --run]` | diagnose, propose, run the next experiment |
| `byosm exp create <name> <exp> --set path=value` · `byosm promote <name> <exp>` | one-line ablations; choose what ships |
| `byosm suite add` · `byosm baseline add` | add a benchmark or a rival |
| `byosm distill <name> [--run]` | shrink a winner into a smaller student |
| `byosm export` · `byosm predict` · `byosm serve` | ONNX bundle, run it, local HTTP API |
| `byosm publish <name> --repo <id>` | push to Hugging Face with a generated model card |
| `byosm bench <a> <b> ...` | specialists vs one generalist (not run yet, see above) |
| `byosm report [--check]` | regenerate the tables in this README |

## Details

**Modalities.** `text`, `audio` (streaming, 10 ms frames), `image`, `document` (PDF or scan: text layer or OCR plus
the page image). Tasks: `classify` and `frame`.

**The bundle.** `byosm export` writes `model.onnx`, `manifest.json` and `byosm_rt.py`. The runtime needs only `numpy`
and `onnxruntime` and gives every modality the same `predict`, `encode`, `stream().step` and `serve`.

**The LLM.** It writes specs, writes and judges text data, analyses errors, and is the `llm` rival. Each role is set in
`byosm.yaml` and falls back to `default`. Here that's local Qwen 3.8 through Ollama:

```yaml
llm:
  default: {provider: openai, model: qwen3.8:27b, base_url: http://localhost:11434/v1, api_key: ollama, reasoning_effort: none}
```

`provider: openai` means any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenAI, OpenRouter).

**Extending.** A modality is a class in `byosm/modalities/` with `@register`. A rival is a `Predictor` with
`@register_baseline`, or any `hf:<repo>` classifier in `baselines:`. An improve rule is a function with `@rule`. An LLM
provider is `register_provider`.

**Layout.**

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
  report/       README tables
specialists/<name>/   spec.yaml, experiments/, baselines/, board.json
docs/design-notes.md  why things are the way they are
```

Datasets, weights and exports are never committed.

**Licence.** CC BY-NC 4.0 ([LICENSE](LICENSE)), for the code and the published models. The VAD's training data
includes ESC-50 (CC BY-NC): check data licences before any other use.
