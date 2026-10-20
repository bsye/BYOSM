# BYOSM: Build Your Own Specialist Model

**Describe a perception task in one sentence. Get a tiny, measured model that does exactly that task and runs on any
computer.**

**One command:**

```bash
byosm auto doctype "classify business documents (PDF or scan) into invoice, letter, memo, form, ..." --budget 6h
```

The autopilot compiles the spec, **discovers** a public benchmark and existing models to beat (and always adds the
zero-shot **LLM** as a baseline), builds experiments until the specialist **beats every baseline** and meets the target
(or the budget ends), distills a large winner into a tiny student if the student still wins, and exports it. Every
decision and its evidence lands in `specialists/<name>/auto.md`. The same steps can also be run by hand:

```bash
byosm new vad "voice activity detection for a voice agent; ignore coughs, laughter and keyboards"
byosm build vad        # data → self-supervised pretraining → training → evaluation → promotion
byosm board vad        # how it compares to the alternatives, on real test sets
byosm improve vad      # what to change next, and why
byosm export vad       # a 0.4 MB ONNX bundle that needs only numpy + onnxruntime
byosm publish vad --repo you/byosm-vad    # Hugging Face: bundle + model card with eval, comparison, lineage
```

---

## Why

Fast, intuitive, *System 1* perception (is someone speaking? did they finish their turn? is this message toxic?) is
usually handled by one general model, or by a large LLM called for every decision. BYOSM bets the other way: **many small
specialists, each trained for exactly one job on exactly its own data, beat one generalist**. They're more accurate on
their task, a fraction of the size, and fast enough to run side by side on a laptop CPU.

Each specialist is a small encoder (JEPA‑style: it learns to *represent* its inputs, not to generate them) with a tiny task
head on top. BYOSM automates everything around that model: defining the task, getting data, training, honest evaluation
against real competitors, deciding what to fix next, and shipping something you can run anywhere.

---

## How it works

```
 "voice activity detection for a voice agent…"
        │
   ①  DEFINE      spec.yaml ◀─ you review / edit
        │           labels + definitions, objective, data plan, backbone, decision policy, eval suites, baselines
        │
   ②  DATA        data/<hash>/  train · dev · test · unlabeled
        │           LLM-written text (double-labelled) · Hugging Face datasets · procedural audio · real labelled audio
        │
   ③  PRETRAIN    LeJEPA self-supervised adaptation on unlabeled in-domain inputs
   ④  TRAIN       tiny head (+ backbone fine-tune), checkpoint chosen on dev
        │
   ⑤  EVALUATE    every eval suite × (every experiment + every baseline), same scorer, same calibration
        │
   ⑥  BOARD       leaderboard → best experiment on the primary suite is promoted
        │
   ⑦  IMPROVE     diagnosis → proposed spec edits (with evidence) → new experiment → back to ②
        │
   ⑧  SHIP        export/: model.onnx + stream.onnx + manifest.json + byosm_rt.py  →  encode / predict / stream / serve
   ⑨  PUBLISH     Hugging Face repo: bundle + generated model card (test eval, baseline comparison, lineage, limitations)
```

### ① Define: the spec

`byosm new <name> "<sentence>"` sends your sentence to an LLM acting as a *task compiler*, which writes
`specialists/<name>/spec.yaml`. The spec is the single source of truth, and you're expected to read and edit it:

- **Labels with definitions** that settle borderline cases ("a cough is non‑speech"; "a 40 ms gap between words is
  still speech"). Vague labels cap precision, so this is where most quality is won.
- **Objective**: what the deployer actually cares about, e.g. `f1`, `macro_f1`, or `recall_at_precision` with
  `constraint: 0.99` (as many detections as possible while keeping 99% precision).
- **Hard cases and diversity axes**: the confusers a naive model gets wrong, and the dimensions real inputs vary along
  (speaker, noise, domain, language…). Both drive data generation.
- **Backbone**: the smallest model from the catalog that can plausibly hit the target within the latency budget.
- **Data plan, decision policy, eval suites, baselines**: covered below.

### ② Data: build the training set

The data plan is turned into `train / dev / test / unlabeled` splits. Every source is optional; mix as needed.

| source | how it works |
|---|---|
| **LLM-written text** | A *generator* prompt writes batches of examples for each label under sampled diversity axes and hard cases, including near‑misses labelled with their true label. A separate **judge** relabels everything blind, and disagreements are dropped. The **dev and test** sets come from an independent *evaluator* prompt (ideally a different model family), deduplicated against train with 5‑gram overlap so they can't leak. |
| **Hugging Face datasets** | The LLM searches the Hub from the spec's queries, picks datasets, and maps their columns and label values onto your labels. Values that don't map cleanly are dropped, because precision beats coverage. |
| **Procedural audio** | Clean positive events (e.g. LibriSpeech speech) are placed over negative backgrounds (ESC‑50 coughs, laughter, rain, keyboards…) at random SNRs. Frame labels come from the clean event's own energy envelope, so they're exact by construction. `fill_ms` sets the label convention (60 ms = word-level, 300 ms = utterance-level). |
| **Real segment-labelled audio** | Repos of recordings with onset/offset sidecars (e.g. AVA‑Speech), chopped into training clips with frame labels. |
| **Local files** | `local_dir/<label>/*`. |

The data directory is named by a **hash of the data plan**. Experiments that only change training settings share the same
data; changing the data plan builds a new set.

### ③ Pretrain: LeJEPA

JEPA (Joint‑Embedding Predictive Architecture) models learn by predicting the *representation* of one view of an input
from another view, rather than reconstructing pixels or samples. That teaches them what's invariant and meaningful, which
is exactly what a perception model needs. The classic failure mode is **collapse**: every input maps to the same vector,
which makes prediction trivially perfect.

BYOSM uses **LeJEPA** (Balestriero & LeCun, 2025), which removes the usual anti-collapse tricks (EMA teachers,
stop‑gradients) and replaces them with one principled regulariser:

- Each unlabeled input produces **2 global views** (most of the input) and **4 local views** (small crops), made by the
  modality plugin's augmentations: text masking and crops; audio crops, gain, noise and time masks; image crops, flips and
  colour jitter.
- **Prediction loss**: every view's embedding is pulled toward the mean embedding of the global views.
- **SIGReg**: the batch's embeddings are projected onto 256 random directions. Along each one, the *empirical
  characteristic function* is compared with a standard Gaussian's (an Epps–Pulley test). Minimising it pushes the
  embedding distribution toward an isotropic Gaussian, which provably can't collapse.
- `loss = (1 − λ)·prediction + λ·SIGReg` with λ = 0.05. BYOSM logs the embedding spread (`emb_std`) every few steps so
  collapse is visible immediately.

This matters most for backbones trained from scratch (`tiny-crnn`, `tiny-cnn`). Pretrained Hugging Face encoders get a short
adaptation pass.

### ④ Train

A tiny head (LayerNorm → Linear) goes on top of the backbone: mean- or attention-pooled for `classify` tasks, per position
for `frame` tasks (one label per 10 ms of audio or per word). Training is class‑weighted cross-entropy with a cosine
schedule and early stopping, and the checkpoint is chosen on the **dev** set. For text, a **refine loop** can run: an
analyst LLM reads the dev errors, names the systematic failure patterns, writes targeted training data for them (judged
and deduplicated like everything else), and the model retrains.

Fine-tuning a pretrained encoder uses one fixed recipe. Each part has a switch in `train:`, and the defaults are what
measured best (see [the recipe, measured](#the-training-recipe-measured)):

| | default | what | why |
|---|---|---|---|
| parameter groups | on | new layers (head, projections, fusion) at `lr_head`; pretrained encoders at `lr_backbone` | layers trained from scratch need a far higher LR than pretrained ones |
| EMA weights | on | an exponential moving average over ~half an epoch is what's selected and shipped | smooths SGD noise at no cost: **+0.5 pt** |
| layer-wise LR decay | floor 0.5 | encoder LR falls with depth, to `layer_decay_floor` × at the embeddings | low layers hold general features worth keeping; a floor of 0.1 was too strong for small encoders |
| mixed precision | auto | fp16 (MPS) / bf16 (CUDA), only when a pretrained encoder is fine-tuned | no accuracy cost; 2.1× faster per step on DiT-base + MiniLM (no gain on a 22M encoder alone) |
| linear probe first | off | `probe_epochs` with the encoders frozen (in eval mode) | didn't help in-domain; available for out-of-domain transfer |

Backbones trained from scratch (`tiny-crnn`, `tiny-cnn`) keep plain fp32 at `lr_backbone`, with EMA.

When the primary suite is external (a public benchmark or your own files), the builtin test split isn't needed for
anything, so it is folded into training (`train.holdout_test: auto`): about 20% more labelled data, for free.

### ⑤ Evaluate: eval suites and baselines

An **eval suite** is a named, frozen test set with a `calib` part and a `test` part:

- `builtin`: the specialist's own adversarial dev/test.
- `hf`: a real benchmark from Hugging Face (rows, or segment-labelled audio such as AVA‑Speech).
- `local`: your own jsonl.

Items from Hugging Face sources are assigned to `train / calib / test` by a **stable hash of their file name**. The same
dataset can therefore be a training source (train partition) and an eval suite (calib + test) with zero overlap, no matter
the order you add them in.

A **baseline** is any competitor: `silero` (LiveKit's VAD, using the exact ONNX file LiveKit ships), `energy` (a sanity
floor), or `llm` (the zero‑shot LLM itself, the "System 2" bar for text). Experiments and baselines go through **the same
scorer**:

1. Run on the suite's `calib` part and **calibrate the decision policy** on it: a threshold, or for streaming audio the
   LiveKit-style **hysteresis** (smoothing, start/stop thresholds, 50 ms minimum speech, 550 ms minimum silence). The
   calib score that selection uses is **cross-fitted**: each half of calib is decided with the policy fitted on the other
   half, so a model doesn't look better just because its threshold was tuned on the items it is scored on.
2. Run on the `test` part and report the objective with a **95% bootstrap CI**, precision/recall, threshold-free ROC‑AUC and
   average precision (binary tasks), parameters, and streaming CPU cost per chunk.

Per-item outcomes are kept (one confusion matrix per file for frame tasks, since frames of a file aren't independent). That
makes every comparison a **paired bootstrap**: "B beats A with P = 0.97" instead of "B is 0.4 points higher".

Results are cached by a fingerprint of the suite, so re-scoring is free until something changes.

### ⑥ Board and promotion

`byosm board <name>` is the leaderboard: experiments × baselines × suites. The experiment with the best objective on the
primary suite's **calib** split is promoted, and promotion decides what `export` and `serve` use. You can override it with
`byosm promote`.

Selection never looks at test. Promotion, the autopilot's keep/revert, its stopping rule and the int8 check all decide on
calib; test is only reported. Otherwise, after ten experiments, the "best" one is partly the one that got lucky on the
test set.

### ⑦ Improve

`byosm improve <name>` reads the board and proposes the next experiment as **explicit edits to the spec**, each with the
evidence behind it:

It first prints a diagnosis: per-suite objective, the gap to the best baseline, the dev training curve, the weakest
labels and the most confused label pairs. Then each rule proposes edits only when the evidence supports them:

| rule | triggers when | proposes |
|---|---|---|
| `real-data-<suite>` | a real-world suite lags (behind a baseline or the builtin suite), and the model never trained on that source | train on the source's train partition (hash-disjoint from the suite's files) |
| `more-data` | losing to a baseline or below target | ~4× more rows from the labelled source (or 2× synthetic), **capped at what the split actually has**. The explanation says whether the curve was still rising (data-limited) or had plateaued. |
| `new-split` | `more-data` would help but the split is used up | add the dataset's largest split that no eval suite reads (e.g. RVL-CDIP `train`, 320k rows, while training on `val`) |
| `regularize` | dev peaked early and never recovered while the training loss kept falling (overfitting) | half the encoder LR; for fused documents, drop a view more often |
| `domain-encoder` | losing to a baseline with an encoder pretrained on another domain (e.g. photos for documents) | a domain-pretrained encoder (e.g. `microsoft/dit-base` for document scans) |
| `train-longer` | the best dev epoch is in the last quarter of a run that wasn't early-stopped | ~1.6× more epochs |
| `utterance-labels` | frame audio, and calibration pushed the threshold to an extreme (label-convention mismatch) | change how training labels are made (`fill_ms: 300`) |
| `hard-pairs` | systematic confusions, with LLM-written data | those pairs as hard cases + an extra refine round |
| `refine-more` | text dev score below target | more error-driven data rounds |
| `llm-*` | an LLM is configured (role `analyze`) | its own proposals, given the board, curve and confusions, limited to allowed spec paths |

Edits address list items by index (`data.hf_sources.0.per_label`). Adding a rule means writing one function decorated
with `@rule` in `byosm/improve.py`.

Every proposal comes with an **estimated cost**: ingest time for the new data (OCR is ~0.36 s per new page, zero for pages
already in the OCR store) plus training time scaled from the last run. The autopilot uses it to skip proposals that don't fit the
remaining budget.

`byosm improve <name> --apply a,b --run` writes `experiments/vN/spec.yaml` and builds and scores it. Every experiment
records which proposal created it.

### ⑧ Ship: runs on most computers

`byosm export <name>` writes a self-contained folder that needs **no PyTorch and no GPU**, just
`pip install numpy onnxruntime` (+ `tokenizers` for text, Pillow for images), on macOS, Linux or Windows:

```
export/vad/
  model.onnx       inputs → embedding (the learned representation) + logits
  stream.onnx      stateful streaming step (causal audio backbones)
  manifest.json    labels, calibrated decision policy, preprocessing config, metrics per suite, provenance
  byosm_rt.py      the runtime: one file
  mel_fb.npy, window.npy | tokenizer.json
```

Preprocessing (log-mel spectrogram, tokenisation, image normalisation) runs in numpy or `tokenizers`, so the ONNX graph
is just the network. Before writing the bundle, export checks it against PyTorch, and streaming against offline inference
(max difference ≈ 5·10⁻⁵).

**The API is the same for every modality:**

```python
from byosm_rt import load
m = load("export/vad")

m.encode(x)        # the representation: [D] pooled, or [T, D] per frame (feed it to probes, retrieval, other heads)
m.predict(x)       # {"label", "probs"}  |  {"segments": [{"label", "start_s", "end_s"}], "frames", "probs"}

s = m.stream()     # realtime: feed chunks of any size (float32 mono, 16 kHz)
s.step(chunk)      # {"probs": [...per 10 ms...], "speaking": bool, "events": [{"type": "start"|"end", "t_s": ...}]}
```

**Or as a local HTTP server**, with many specialists in one process:

```bash
python export/vad/byosm_rt.py serve export/vad export/eot --port 8800
```

```
GET  /v1/models                       → manifests (labels, metrics)
POST /v1/vad/predict  {"input": "clip.wav"} | {"pcm16_b64": "..."}
POST /v1/vad/encode   {...}           → {"embedding": [...]}
POST /v1/predict      {...}           → every model's answer at once
```

### ⑨ Publish to Hugging Face

```bash
huggingface-cli login                      # once
byosm publish vad --repo you/byosm-vad --license cc-by-nc-4.0 [--public] [--dry-run]
```

This pushes the exported bundle plus `spec.yaml`, `board.json` and `lineage.json`, and a **model card generated from the
project's own artifacts**, so it can't drift from what was measured:

- **What it does**: description, labels and definitions, size, decision policy.
- **Comparison**: every experiment and baseline on every suite, scored identically.
- **Test evaluation**: per-suite objective, precision/recall, ROC‑AUC, AP and calibrated threshold, also written as
  `model-index` metadata, which the Hub renders as an evaluation widget.
- **How it was built**: data sources, LeJEPA, training, and the `improve` lineage with the evidence behind each step.
- **Limitations**, generated automatically: in-domain caveats (trained on a partition of an eval source), non-commercial
  datasets, small suites.
- **Usage**: `snapshot_download` + `byosm_rt`, streaming, HTTP server.

Repos are private unless you pass `--public`. `--dry-run` stages everything under `specialists/<name>/hub/` for review.

---

### Autopilot: how it decides

| stage | decision | rule |
|---|---|---|
| define | task, labels, objective, backbone | spec compiler (LLM) from the prompt |
| discover | primary eval suite | a public benchmark found on the Hub (its held-out split), with its train split as training data |
| discover | baselines | the zero-shot LLM (`llm`, or `llm:<role>` for each extra LLM in `byosm.yaml`; multimodal for documents and images) + existing Hub models whose labels cover ≥60% of ours |
| loop | next experiment | among untried `improve` proposals that fit the remaining time budget, the highest prior (domain data/encoder > label convention > more data / new split > regularize, hard pairs > train longer), cheaper first on ties; **one change per round** so every gain is attributable |
| loop | keep or revert | promoted only if it beats the current best on the primary suite's calib split; it counts as progress only if a paired bootstrap gives P(better) ≥ 0.9 |
| stop | when | **won** (beats every baseline by `--margin` on calib and meets the target), time or experiment budget, 2 rounds without a significant gain, or nothing left to try |
| distill | shrink? | if the winner has more than `--max-params` M parameters: distill into a tiny student (soft targets on train + unlabeled inputs); promoted only if it still beats every baseline |
| ship | export / publish | ONNX bundle with int8 weights: tries MatMul/Gemm + embedding tables, then MatMul/Gemm only, and ships the first that costs ≤ 0.5 pt on calib (else fp32); `--publish <repo>` pushes it with a model card |

## The training recipe, measured

Banking77 (77 intents, official test split, 2,460 items after the calib split), MiniLM-L6 (22M), 3 seeds each, same data:

| recipe | test accuracy (mean ± sd) | vs old |
|---|---|---|
| old: one LR for the encoder, fp32, last-step weights | 92.52 ± 0.21 | |
| + mixed precision | 92.53 ± 0.25 | +0.01 |
| + mixed precision + EMA | 93.01 ± 0.17 | +0.49 |
| + mixed precision + layer decay 0.5 | 92.59 ± 0.34 | +0.07 |
| **+ mixed precision + EMA + layer decay 0.5 (default)** | **93.14 ± 0.17** | **+0.62** |
| + linear probe epoch | 92.97 ± 0.24 | +0.45 |
| **default + builtin test split folded into training** (`holdout_test: auto`) | **93.44 ± 0.22** | **+0.92** |

An earlier round with a layer-decay floor of 0.1 and a probe epoch was worse than this default on every seed, which is
why the floor is 0.5 and the probe is off.

## Distillation

`byosm distill <name> [--teacher exp] --run` trains a tiny student on `α·CE(gold) + (1−α)·T²·KL(teacher ‖ student)`,
over the training inputs plus up to `distill_unlabeled` extra unlabeled inputs that only the teacher labels. Default
students are small *pretrained* encoders, which beat from-scratch ones on small data: tiny-BERT (text, 11M), tiny-crnn
(audio, 0.1M), DeiT-tiny (image, 5.7M), tiny-BERT + DeiT-tiny (document, ~17M).

On RVL-CDIP the 17.6M document student keeps 97% of its 110M teacher's accuracy at 1/6 the size and 4× the speed:

| | params | RVL-CDIP test | CPU/doc (PyTorch) | ONNX bundle (int8, calib accuracy unchanged) |
|---|---|---|---|---|
| teacher `r2` (MiniLM + DiT-base) | 109.6M | 90.5% | 48 ms | 438 → **113 MB** (incl. embeddings) |
| student `distill-r2` (tiny-BERT + DeiT-tiny) | 17.6M | 88.2% | 11 ms | 70 → **43 MB** (embeddings stay fp32: quantising them cost 2 of 384 docs in the int8 check) |

## Results: a 0.1M-param VAD vs LiveKit's Silero

Published with `byosm publish`: **[huggingface.co/bsye/byosm-vad](https://huggingface.co/bsye/byosm-vad)** (model card, eval widget, ONNX bundle).

Frozen AVA‑Speech suite: 20 real film/TV recordings, ~5 h, labelled by humans, never trained on. Every model goes
through LiveKit's own hysteresis logic, calibrated on 8 separate recordings, on one CPU thread.

| | params | CPU per chunk | builtin F1 | **AVA F1** | AVA AUC |
|---|---|---|---|---|---|
| **BYOSM vad v3** (v2 + new trainer) | **0.10M** | 0.070 ms / 30 ms ¹ | – ² | **0.882** | **0.942** |
| BYOSM vad v2 | 0.10M | 0.070 ms / 30 ms ¹ | 0.879 | 0.876 | 0.939 |
| Silero (LiveKit) | 0.55M | 0.068 ms / 32 ms | 0.885 | 0.865 | 0.923 |
| BYOSM vad v1 | 0.10M | 0.070 ms / 30 ms ¹ | 0.894 | 0.849 | 0.900 |
| energy floor | – | – | 0.728 | 0.728 | 0.694 |

¹ the exported ONNX bundle (`byosm board` reports the slower PyTorch timing). ² v3 trains on the builtin test split
(`holdout_test: auto`), so it isn't scored there. The published model is v2.

**How v1 became v2.** v1 was trained only on synthetic mixtures (LibriSpeech over ESC‑50) and lost to Silero on real audio.
`byosm improve` read the board and proposed two changes. `real-data-ava`: the model had never heard film audio, so train
on AVA's train partition (hash-disjoint from the test files). `utterance-labels`: calibration pushed the threshold down to
0.20, a sign that AVA labels whole utterances while our labels covered only voiced frames. Applying both produced v2: same
size, same speed, ahead of Silero on real audio.

**The caveat:** v2 has seen film audio and Silero hasn't. This is an *in‑domain* win, which is exactly the thesis (tailor
the model to its domain), not a claim that v2 is better everywhere. It's also slightly *worse* on the synthetic suite,
whose labels use the convention v2 was moved away from.

## Documents: PDFs and scans, with multimodal fusion

`modality: document` treats a PDF (or a page image) as a list of **pages**. Each page has two views:

- **text**: the PDF's text layer, or OCR for scanned pages (RapidOCR, ONNX). OCR text goes into a content-addressed
  store (`~/.cache/byosm/ocr`, or `$BYOSM_CACHE`), keyed by a hash of the page's pixels and the OCR version, so a page is
  OCR'd once per machine, across every specialist and data rebuild. Uncached pages run one single-threaded OCR engine
  per core: 0.26 s/page on an 18-core M5 Max, 7× faster than one engine using all cores.
- **visual**: the rendered page image

`document.route` picks `text`, `visual` or **`fused`**. Fused pages combine both views with a learned per-dimension gate,
and a random view is dropped during training so a page with no text, or a blank render, still works. A small attention
block then mixes information across pages, and attention pooling decides the document label. `frame` tasks label every
page instead (e.g. finding where one document ends and the next begins in a merged scan).

RVL-CDIP (16 document types), scored on 3,200 documents from the **original test split** (95% CI about ±1 pt; another
800 from that split form the calib part that selection uses):

| | params | accuracy | CPU/doc | trained on |
|---|---|---|---|---|
| DiT-base (`microsoft/dit-base-finetuned-rvlcdip`, baseline) | 85.8M | 94.0% | 37 ms | 320k labelled + 42M unlabelled documents |
| **BYOSM v2-ditft** (v1 with its visual branch started from DiT-base's fine-tuned weights) | 109.6M | **94.8%** | 46 ms | 34,935 documents, on top of DiT's 320k |
| BYOSM v1 (r2 + all of the val split, *proposed by `byosm improve`*) | 109.6M | 91.8% | 45 ms | 34,935 documents (val split + builtin test) |
| BYOSM r2 (fused-dit + new trainer + builtin test folded in) | 109.6M | 90.5% | 48 ms | 11,520 documents |
| BYOSM fused-dit (MiniLM text + DiT-base visual, *proposed by `byosm improve`*) | 109.6M | 88.4% | 47 ms | 9,600 documents |
| BYOSM distill-r2 (tiny-BERT + DeiT-tiny student of r2) | 17.6M | 88.2% | 11 ms | same as r2 + r2's soft labels |
| BYOSM fused (MiniLM text + DINOv2-small visual) | 45.7M | 80.1% | 24 ms | 2,400 documents |
| BYOSM text (OCR → MiniLM) | 23.4M | 74.1% | 7 ms | same |
| BYOSM visual (DINOv2-small) | 22.8M | 64.5% | 19 ms | same |

`fused` → `fused-dit` applied three proposals from `byosm improve`: 4× data, a document-pretrained visual encoder
(`microsoft/dit-base`, self-supervised on unlabelled scans) and longer training. The new trainer (EMA, layer decay,
fp16) and folding in the builtin test split gave r2; `more-data` (every val document, ~2,400 per type) gave v1. The gap
to DiT went 14 → 6 → 3.5 → **2.2 points**, with 9× fewer labelled documents than DiT was fine-tuned on.

**Beating DiT-base: add the view it lacks.** v1 lost to DiT on visually distinctive types (news articles, scientific
reports), but was right on 64 test documents DiT got wrong, because it reads the text. v2-ditft keeps v1's text branch
and fusion and starts the visual branch from DiT-base's own fine-tuned checkpoint (`document.image_pool: pooler` reads it
exactly as DiT's classifier did). On the 3,200 test documents it beats DiT by **+0.8 pt** (paired bootstrap 95% CI +0.3
to +1.3, P = 1.00; fixes 50 of DiT's errors, adds 24). On the 800 calib documents the two tie (93.5% vs 93.6%), so the
autopilot would not yet call it a win. Dev accuracy peaked at epoch 2 and then fell, which suggests the visual branch is
drifting from what DiT learned; a lower visual learning rate is the next experiment. The claim is "DiT + text beats
DiT", not "beat DiT from less data": DiT's checkpoint brings its 320k labelled documents. RVL-CDIP's test split is
disjoint from the train split DiT was fine-tuned on.

Fusion adds 6 points over the best single view, because the views fail on different classes. Visual is strong on
`handwritten`, `advertisement` and `file_folder`; text is strong on `questionnaire`, `memo` and `invoice`.

The first, 768-document version of this suite had a 192-document calib part, on which v1 and r2 tied exactly despite a
2-point test gap: too small to select on. Benchmark suites now default to ~4,000 items (~800 calib). The next step
`byosm improve` proposes is `new-split`: the val split is used up, and RVL-CDIP's 320k-document `train` split is unused.
A clean, born-digital invoice PDF is also out of distribution for a
model trained on 1990s scans (it predicted `budget` 0.73 / `invoice` 0.27); add an eval suite of your own PDFs to see
that gap on the board.

```bash
byosm exp create doctype fused --set document.route=fused     # ablations are one line each
byosm build doctype --exp fused
python specialists/doctype/export/byosm_rt.py predict specialists/doctype/export invoice.pdf
curl -X POST localhost:8800/v1/doctype/predict -d '{"pdf_b64": "..."}'
```

---

## Quickstart

```bash
git clone https://github.com/bsye/BYOSM && cd BYOSM
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev,silero]'
byosm init                                   # writes byosm.yaml: which LLM to use for each role
byosm new eot "decide if a voice-agent user finished their turn or is mid-thought; en/es/it"
byosm build eot
byosm board eot
byosm export eot && python specialists/eot/export/byosm_rt.py predict specialists/eot/export "so what I wanted to ask was"
```

The VAD example is reproducible without any LLM. `specialists/vad/spec.yaml` lists every data source explicitly:

```bash
byosm build vad && byosm board vad
```

### LLMs: any provider, per role

An LLM is only used to compile specs, write and label text data, analyse errors, and serve as the `llm` baseline. Each role
is configured in `byosm.yaml`:

```yaml
llm:
  default:  {provider: anthropic, model: claude-opus-5-5, effort: medium}     # ANTHROPIC_API_KEY
  generate: {provider: openai, model: qwen3:14b, base_url: http://localhost:11434/v1, api_key: ollama}
  eval:     {provider: openai, model: gpt-5, api_key_env: OPENAI_API_KEY}     # different family for held-out data
  # spec / judge / analyze / baseline fall back to default
```

`provider: openai` means any OpenAI-compatible server: OpenAI, Ollama, vLLM, LM Studio, OpenRouter… You can register
anything else with `byosm.llm.register_provider(name, factory)`.

### Commands

| | |
|---|---|
| `byosm auto <name> "<task>" [--budget 6] [--margin 0] [--publish repo]` | **everything, decided automatically** |
| `byosm new <name> "<task>"` | compile a spec from a sentence |
| `byosm suite add <name> <suite> --hf <id> [--format segment_files] [--primary]` | add a frozen eval suite |
| `byosm baseline add <name> silero\|energy\|llm` | add a competitor |
| `byosm build <name> [--exp vN]` | data → pretrain → train → evaluate → promote |
| `byosm eval <name>` · `byosm board <name> [--detail]` | score and compare |
| `byosm improve <name> [--apply ids --run]` | diagnose, propose, run the next experiment |
| `byosm promote <name> <exp>` | choose what ships |
| `byosm distill <name> [--teacher exp] [--run]` | shrink a teacher into a tiny student |
| `byosm export <name>` · `byosm serve <names…>` · `byosm predict <name> <input>` | ship and run |
| `byosm publish <name> --repo <id> [--public] [--dry-run]` | push to Hugging Face with a generated model card |
| `byosm bench <a> <b> …` | the thesis test: N specialists vs one shared-backbone generalist on the same data |

---

## Extending

- **New modality**: one file in `byosm/modalities/` implementing `load`, `build_backbone`, `collate`, `views` (the JEPA
  augmentations) and `example_input`, decorated with `@register`. Training, LeJEPA, evaluation and export are all
  modality-agnostic.
- **New baseline**: subclass `Predictor` in `byosm/predictors.py` with `@register_baseline("name")`, or use any
  Hugging Face classifier directly as `hf:<repo>` (its labels are matched to yours by name).
- **New LLM provider**: `register_provider`.

## Layout

```
byosm/
  spec.py          TaskSpec, spec compiler, backbone catalog
  llm.py           provider-agnostic structured-JSON LLM calls
  data/            synth_text · hf_source · audio_mix · segments · store (dedupe, leakage control)
  modalities/      text · audio (tiny-crnn, streaming) · image
  jepa.py          LeJEPA (multi-view prediction + SIGReg)
  model.py         backbone + heads + projector
  train.py         supervised training, dev selection
  predictors.py    experiments and baselines behind one interface
  evals.py         suites, scoring, calibration, board
  improve.py       diagnosis and proposals
  pipeline.py      experiment pipeline
  export.py        ONNX bundle + parity check
  rt.py            portable runtime (copied into every bundle as byosm_rt.py)
  auto.py          autopilot: define → discover → experiment loop → distill → ship
  discover.py      benchmark + baseline discovery on the Hub
  distill.py       teacher → tiny student
  hub.py           Hugging Face publishing + model card
  bench.py         specialists vs generalist
benchmarks/        standalone comparisons (vad_vs_silero.py)
specialists/<name>/spec.yaml, experiments/*/{spec.yaml,meta.json,results/}, board.json
```

Datasets, model weights and exports aren't committed. The VAD's training data includes ESC‑50 (CC BY‑NC), so check data
licences before shipping a model commercially.

## Status

Working end to end: audio (the VAD above), documents (RVL-CDIP above, 94.8%, ahead of DiT-base on test), and text on a real benchmark (Banking77, 93.4%
with a 22M encoder). The autopilot's LLM-driven steps (spec compiling, synthetic data, the `llm` baseline) are tested with
a stub; a real `byosm auto` run and the `bench` thesis experiment need an LLM key or a local model.
