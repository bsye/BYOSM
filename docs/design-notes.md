# Design notes

The code has no comments. This file holds the reasons behind the design: why a choice was made, what was measured, and
the gotchas. Each entry names the module it is about. Numbers are either constants in the code or results in
`specialists/*/board.json`.

## Spec

**The spec is the single source of truth.**
`byosm/spec/task.py`: `TaskSpec` is compiled from one sentence and then edited as YAML. Every experiment keeps a snapshot
of it, so an experiment can be rebuilt from its own `spec.yaml`. Gotcha: scoring and promotion read the base spec, not
the experiment's: its `decision`, `objective`, `evals`, `baselines`, labels and `positive_label`
(`byosm/evals/board.py`, `byosm/pipeline.py` `maybe_promote`, `byosm/auto/standing.py`). Editing those in an
experiment's `spec.yaml` changes nothing on the board.

**One directory per specialist.**
`byosm/config.py` `Project`: `specialists/<name>/` holds the base `spec.yaml`, `data/<data_key>/`, `evals/<suite>/`
(frozen suites), `baselines/<id>/` and `experiments/<exp>/` (model, snapshot spec, `meta.json`, `results/`), plus
`board.json` and a `promoted` file naming the shipped experiment. Only directories with a `meta.json` count as
experiments.

**Discovery turns a public benchmark into the primary suite.**
`byosm/discover.py` `_add_benchmark`, `_split_sources`: when a matching Hub dataset is found, its held-out split becomes
a balanced `hf` suite named `bench` and the new primary (every other suite stops being primary), and its `train` split,
if there is one, is added as a balanced training source. Default baselines: `llm` for text, document and image
classification; `energy` and `silero` for frame-level audio; plus Hub classifiers found by search. Gotcha: a primary
suite that is not builtin turns on `holdout_test: auto` folding (see below), so discovery also decides that the builtin
test split goes into training.

**`data_key` decides when data is shared.**
`byosm/spec/task.py`: the key hashes everything that shapes the dataset, so experiments with equal keys reuse the same
`data/<key>/`. Label definitions and the task description count only when `data.synthetic` is on, because only then
does an LLM read them to write data. Editing a definition of a real-data spec does not throw the cached data away.
Gotcha: refine rounds append to that shared `train.jsonl`, so a later experiment with the same key inherits them.

**The backbone catalog is a menu, not a limit.**
`byosm/spec/backbones.py`: `BACKBONES` is what the compiler may choose from; any `hf:<repo>` works when set by hand. An
unknown id from the LLM falls back to `DEFAULT_BACKBONE`: the smallest entry per modality, except text, which gets
MiniLM-L6 instead of tiny BERT.

**Hash partitions keep training and evaluation apart, where they are used.**
`byosm/data/segments.py`: `partition_of` puts each item in train (70%), calib (6%) or test (24%) by an md5 of its key.
`segment_files` suites read the calib and test partitions. A `segment_files` training source with `partition: train` or
`all` reads train and does not overlap them; any other partition is read as given, so `calib` or `test` overlaps the
suite (`byosm/data/__init__.py` `_segment_clips`). Balanced parquet sampling (training sources and `hf` suites alike,
including discovery's `bench` suite) also filters rows by the md5 partition of `<file>:<row>`. Gotchas:
`sampling: stream` ignores `partition`, and an `hf` suite then sends every fifth row it takes to calib (see "Suites are
frozen on first use").

**Balanced sampling exists for label-sorted datasets.**
`byosm/data/hf_source/parquet.py`: RVL-CDIP and similar datasets are sorted by label, so reading a stream in order only
ever sees the first labels. `balanced_rows` reads only the label column of the split's parquet files to build an index,
draws `per_label` random rows per label, then decodes only those rows. Gotcha: every parquet file of the split is
downloaded to build the index, and the whole input column of each file holding a chosen row is read, so for image
datasets such as RVL-CDIP most of the bytes are fetched anyway. Files are cached by `hf_hub_download`.

**`label_map` filters, and its empty form differs per reader.**
`byosm/data/hf_source/rows.py`, `parquet.py`, `byosm/data/__init__.py`: pairs are `[source_value, our_label]` and
unlisted source values are dropped. With an empty map, balanced parquet sampling matches source labels to ours by
normalised name, while stream sampling gives every row the label `None`, so labelled rows are lost. To filter rows
for positive/negative/unlabeled sources, rows sources map wanted values to `keep` (`[value, "keep"]`), but
`segment_files` sources use the reverse order (`["keep", cluster]`).

**Audio frame labels follow the signal.**
`byosm/spec/sections.py` `AudioCfg`: with `label_by_energy`, frames come from the positive clip's own energy, so pauses
inside speech are negative. Gaps shorter than `fill_ms` stay positive: about 60 ms gives word-level labels, about 300 ms
utterance-level labels.

**Only binary tasks get a fitted threshold.**
`byosm/spec/sections.py` `Decision`, `byosm/metrics.py` `_binary`, `calibrate`: `threshold` is the only decision value
that can be left empty. A `None` threshold is fitted on each suite's calib split only for binary tasks: `positive_label`
set, or two labels with objective `f1`, `recall_at_precision` or `precision_at_recall`. Every other task (multi-class,
or two labels with `macro_f1`/`accuracy` and no `positive_label`) decides by argmax and calibrates nothing. The fit
maximises the objective for `recall_at_precision` and `precision_at_recall`, binary F1 otherwise. Hysteresis
activation is fitted separately (see "Hysteresis activation is fitted for F1"); the other hysteresis values are used as
written. Hysteresis for frame tasks follows LiveKit/Silero: smooth, enter
at the threshold, leave below `threshold - deactivation_offset`, and require `min_on_ms` to start and `min_off_ms` of
silence to end.

**`holdout_test: auto` reuses a test split that measures nothing.**
`byosm/spec/task.py` `trains_on_test_split`: when the primary suite is external, the builtin test split is only a
report, so it is folded into training. Those experiments record `n_test_folded` and are skipped on the builtin suite.
Gotcha: `byosm bench` does not check this (see "The thesis benchmark").

## Data

**Held-out text can be written by a different model with a different brief.**
`byosm/data/synth_text.py`, `byosm/llm/__init__.py`: train comes from the `generate` role; dev and test come from the
`eval` role with an adversarial prompt (confusers, near misses). Gotcha: every role falls back to `default`, and the
shipped config sets only `default`, so out of the box one model writes train and test, judges, refines and is the `llm`
baseline graded on test sets it wrote. Point `eval` and `baseline` at other model families in `byosm.yaml` to get
the separation.

**A second labeller removes label noise.**
`byosm/data/synth_text.py` `judge`: the `judge` role relabels each generated row blind, and only rows where both labels
agree are kept. Label noise caps the precision a tiny model can reach, and this is cheap insurance. It is a second
opinion only when `judge` is configured as a different model.

**Refinement learns from dev errors.**
`byosm/pipeline.py` `refine`, `byosm/data/synth_text.py` `refine_jobs`: the `analyze` role reads the dev errors and
returns failure patterns with data recipes; the `generate` role then writes targeted training rows from them (filtered
by `judge` when it is on), deduped against train, dev and test. The refine loop itself never sees test rows. Gotchas:
dev is the builtin suite's calib split, so the builtin selection value is steered by the same errors that refine trains
on; and the `hard-pairs` improve rule feeds test-split confusions into `hard_cases` (see "Promotion decides on calib").

**Text held-out sets are deduped against train.**
`byosm/data/__init__.py`, `byosm/data/store.py`: for text, test is deduped against train+dev and dev against train; a
row is dropped if it is an exact normalised duplicate or its 5-gram Jaccard with any reference row is above 0.7.
Gotchas: no other modality is deduped. The builtin suite always reads the base spec's data, so an experiment with a
different `data_key` (for example after `more-data`) re-splits its own rows and can train on base dev/test rows.

**Procedural audio has exact labels.**
`byosm/data/audio_mix.py`: clean positive events are placed over negative backgrounds at controlled SNRs, so labels are
exact by construction. Pools are split into train/dev/test before mixing, so test clips never reuse train source audio.
About 12% of backgrounds are near silence with mic hiss (-80 to -50 dB).

**A frame is active within 35 dB of the event peak.**
`byosm/data/audio_mix.py` `frame_activity`: active means within `rel_db` of the event's peak and above -55 dB. Gaps up
to `fill_ms` are filled so labels do not flicker inside words.

**Model selection also sees real audio, though not independent of train.**
`byosm/data/__init__.py`: one in ten real segment-labelled clips (for example AVA-Speech) goes to dev instead of train,
so dev is not only procedural mixes. Gotcha: those clips are adjacent windows of the same recordings as the train clips.
A `segment_files` training source with `partition: all` reads only the train hash partition.

**An empty dataset is never cached.**
`byosm/data/__init__.py` `build`: it raises before writing any split when there are no training examples, so a bad data
plan does not leave a `train.jsonl` that later runs would reuse.

**Media is decoded without ffmpeg.**
`byosm/data/hf_source/decode.py`, `hub.py`: audio and images are decoded with soundfile and PIL, and streaming datasets
get their Audio/Image columns cast to `decode=False`, so the `datasets` decoder (and ffmpeg) is never needed.

**ClassLabel names come from the parquet schema first.**
`byosm/data/hf_source/parquet.py` `_class_names`: the `huggingface` metadata in the parquet schema is read when present;
otherwise the dataset builder info is loaded.

## OCR and documents

**A page has two views and the model uses both.**
`byosm/modalities/document/`: each page has a text view (PDF text layer, or OCR for scans) and a visual view (the
rendered page). `route: fused` combines them with a learned per-dimension gate, then a small attention block mixes
pages. On doctype the rvl test objective was 0.7409 text-only, 0.6447 visual-only and 0.8009 fused.

**View dropout drops one view, never both.**
`byosm/modalities/document/model.py` `DocumentBackbone.pages`: during fused training each page loses its text or its
image with probability `modality_dropout / 2` each, never both, so a page with no text or a blank render still works.

**The page mixer is written by hand for ONNX.**
`byosm/modalities/document/model.py` `AttnBlock`: a plain pre-norm self-attention block, because the fast path of
`nn.TransformerEncoder` does not export cleanly to ONNX.

**Document types are decided early.**
`byosm/spec/sections.py` `DocumentCfg`: `max_pages` keeps the first pages. `ocr: auto` runs OCR only on pages whose text
layer has fewer than 20 characters (scans). `image_pool: pooler` uses the encoder's own trained pooling (for BEiT/DiT,
patch mean plus LayerNorm), which is what its fine-tuned head read.

**Only PDF rendering is serialised.**
`byosm/modalities/document/pages.py`, `byosm/rt.py`: PDFium is not thread-safe but RapidOCR is, so a lock covers
rendering only. `Document.load_many` renders in threads, then OCRs every page missing from the store in one batch.

**One single-threaded OCR engine per core.**
`byosm/modalities/document/engine.py` `ocr_many`: the OCR models are small, so intra-op threads mostly synchronise;
one engine per process with one thread each is faster. Batches smaller than `2 x cpu_count` stay in-process because pool
start-up would dominate. A new pool is made per batch: start-up takes seconds, while idle workers would hold several GB
of RAM for the whole training run.

**OCR is cached by pixels, once per machine.**
`byosm/modalities/document/store.py` `OCRStore`: the key is sha256 of the engine version, image mode, size and pixels.
Rebuilt datasets rename media files but keep the pixels, so every specialist and data build shares the store at
`$BYOSM_CACHE/ocr` (default `~/.cache/byosm/ocr`). Writes are atomic (per-pid temp file, then replace), and the pool
stores each page as soon as it is done, so an interrupted run keeps its progress. Gotchas: upgrading
`rapidocr_onnxruntime` changes the key and re-runs OCR, and so does a new machine or an empty store; legacy
`<file>.ocr.json` sidecars are not read by training.

**Runtime OCR differs from training OCR.**
`byosm/rt.py` `_fill_ocr`, `_ocr`: the bundled runtime has no `OCRStore`; it still reads a legacy `<file>.ocr.json`
sidecar when present, and otherwise runs OCR on each scanned page itself. Gotcha: when `rapidocr_onnxruntime` is not
installed, both training (`byosm/modalities/document/engine.py` `ocr`, `ocr_many`) and the runtime silently return empty
text, so scanned pages lose their text view without an error, and a model trained with OCR then serves without it.

## Training

**Pretrained encoders get layer-wise LR decay.**
`byosm/train/common.py` `param_groups`: low layers hold general features worth keeping, so the top layer trains at
`lr_backbone` and the embeddings at `lr_backbone * layer_decay_floor`. Layers trained from scratch (heads, projections,
fusion) get `lr_head`. A backbone with no pretrained encoder (tiny-crnn, tiny-cnn) trains at `lr_backbone`. Depth is read
from parameter names: a `layer|layers|blocks|block|h|stages` segment followed by an integer; names with `embed` are depth 0.

**Mixed precision only for pretrained encoders.**
`byosm/train/common.py` `amp_dtype`: `train.precision: auto` uses fp16 on MPS and bf16 on CUDA (when supported) only
when the model has a pretrained encoder, where it makes each step much faster; small from-scratch recurrent nets stay
in fp32.

**EMA over about half an epoch.**
`byosm/train/trainer.py` `_ema`: decay is `1 - 1/max(10, steps_per_epoch/2)`, which smooths SGD noise without lagging
behind the cosine LR schedule. The EMA weights are the ones evaluated and shipped.

**Probe epochs are capped and frozen encoders are in eval mode.**
`byosm/train/trainer.py`: linear-probe epochs are `min(probe_epochs, epochs // 4)` and are skipped with
`freeze_backbone` or without a pretrained encoder. A frozen encoder is switched to `eval()` so it is a fixed feature
extractor: no dropout, no BatchNorm statistic updates.

**Missing dev split.**
`byosm/train/loop.py` `_split_rows`: a random 10% of train (at least one row) is held out with the run's seeded RNG.

**Epochs are chosen on dev, with early stopping.**
`byosm/train/loop.py` `_select`, `PATIENCE`: after each epoch the dev objective is measured and the best epoch's weights
are snapshotted; ties go to the higher dev macro F1. Training stops after `PATIENCE` (4) epochs without a new best, and
the snapshot is what is saved. Gotcha: the dev threshold is fitted on dev itself (`_dev_epoch` passes `calibrate=True`
to `byosm/train/evaluate.py`), so `meta["dev"]` is optimistic in the way cross-fitted calib scores avoid. Epoch
selection, early stopping and the refine loop's target check in `byosm/pipeline.py` `build` all use that value.

**Loss settings for imbalanced, noisy labels.**
`byosm/train/trainer.py` `class_weight`, `byosm/model.py` `task_loss`: class weights are the square root of inverse
frequency, which helps rare labels without letting them dominate. Cross-entropy uses label smoothing 0.02. The LR
schedule is warmup then cosine, with warmup `steps // 15`.

**LeJEPA domain adaptation needs no teacher.**
`byosm/jepa.py`: each view's embedding is pulled toward the mean of the global views, while SIGReg pushes the batch
toward an isotropic Gaussian along random 1-D projections (an Epps-Pulley test on the characteristic function). No EMA
teacher, no stop-gradient, one weight (`pretrain_lambda`). Each modality only supplies `views()`.

**The prediction term is a mean, not a sum.**
`byosm/jepa.py` `lejepa_loss`: summing over dimensions makes collapse cheaper than SIGReg's penalty. `emb_std` (per-dim
std across the batch) is logged as a collapse monitor; SIGReg drives it toward 1.

**Distillation keeps everything but the backbone.**
`byosm/distill.py`, `byosm/train/teacher.py`: the student keeps the teacher's task, labels, data and decision policy and
trains on `alpha * gold loss + (1 - alpha) * T^2 KL(teacher || student)`. Up to `distill_unlabeled` extra inputs (not
already in train) are labelled only by the teacher and get `IGNORE` as gold. The vision student is DeiT-tiny, because a
pretrained small student beats a from-scratch CNN on small data. Frame-task teachers run on CPU.

**TinyCRNN is causal and streams exactly.**
`byosm/modalities/audio.py` `TinyCRNN`: the waveform is left-padded by one window minus one hop, so mel frame `t` ends at
sample `(t+1)*hop` and lines up with the frame labels. `stream()` carries the waveform tail, per-conv left context and
GRU state, and matches `forward()` on the concatenated audio.

**Frame targets follow the model's frame rate.**
`byosm/model.py` `align_targets`: when label rate and backbone frame rate differ (10 ms labels, 20 ms wav2vec2 frames),
targets are nearest-resampled to the output length. Padding uses `IGNORE`.

**Word-level text labels go on the first sub-token.**
`byosm/modalities/text.py` `collate`: later sub-tokens and special tokens stay `IGNORE`.

**Images use the encoder's own normalisation.**
`byosm/modalities/image.py` `norm_for`: mean/std come from the encoder's `AutoImageProcessor`; ImageNet stats are the
fallback for tiny-cnn or when the processor has none.

## Evaluation and selection

**One scorer for everything.**
`byosm/evals/scoring.py`, `byosm/predictors/base.py`: every predictor, ours or a baseline, returns probabilities over the
spec's labels and is scored by the same code. Calib probabilities fit the decision policy; test probabilities give the
objective, P/R/F1 and, for binary tasks with both classes present, ROC-AUC and AP.

**Every experiment is scored with the base decision policy.**
`byosm/evals/board.py` `_experiment_predictor`: the experiment's own `decision` is replaced by the base spec's, so all
experiments and baselines are compared under one policy. `decision.*` is not an improve path, and a `decision` edit in
an experiment's `spec.yaml` has no effect on its scores.

**Promotion decides on calib, with gaps.**
`byosm/evals/selection.py` `selection_value`, `byosm/pipeline.py` `maybe_promote`, `byosm/auto/standing.py`: promotion,
autopilot keep/revert and the win check compare the primary suite's calib objective. Test is reported with a 95% CI and
a paired bootstrap against the best baseline. Gotchas:
- `selection_value` falls back to the test objective when a result has no `calib_objective`: a `local` suite with no
  `split: calib` rows, an empty calib split, or an old cached result. Selection then decides on test.
- The improve diagnosis (`byosm/improve/diagnosis.py`, `registry.py` `gap`) reads the test objective, so the lag that
  triggers `real-data`, `more-data` and `domain-encoder`, and the weakest labels and confusions shown to the LLM
  analyst, all come from test. Test steers which experiments are built. The `hard-pairs` rule
  (`byosm/improve/rules_fit.py` `confusable_pairs`) goes further: it copies the top test-split confusions into
  `hard_cases`, which synthetic generation then writes training rows for, so test also shapes training data.
- The distillation check compares the student's test objective with the best baseline's calib value (see "Autopilot
  stops for a reason").

**Calib decisions are cross-fitted.**
`byosm/evals/scoring.py`: deciding calib with a threshold fitted on calib would bias the selection value upward. For
hysteresis, even files are decided with the activation fitted on odd files and the other way round; for thresholds the
same is done on even/odd rows (binary tasks with an uncalibrated threshold).

**Hysteresis activation is fitted for F1.**
`byosm/evals/scoring.py` `_fit_activation`: the activation is picked from a grid of 33 values in [0.1, 0.9] by frame F1
on calib, whatever the spec's objective.

**Infeasible thresholds still rank.**
`byosm/metrics.py` `calibrate`: for `recall_at_precision` and `precision_at_recall`, a threshold that misses the
constraint scores `-1 + constrained metric`, so when none is feasible the closest one wins instead of an arbitrary one.

**Bootstrap units respect dependence.**
`byosm/metrics.py` `units`, `bootstrap`: classify stores one (gold, decision) pair per item; frame tasks store one
confusion matrix per file, because frames of one file are not independent. Every objective is a function of the summed
confusion matrix, so resampling units gives CIs and paired tests. `p_better` counts ties as half.

**Suites are frozen on first use.**
`byosm/evals/suites.py` `materialize`: `local` and `hf` suites are written to `evals/<suite>/` (calib, test, suite.json)
and read back afterwards. The builtin suite uses the base spec's dev (calib) and test splits. `local` rows go to calib
only when marked `split: calib`. `hf` rows suites shuffle the rows they take and send every fifth item to calib.

**About 800 calib items for a benchmark.**
`byosm/discover.py` `_split_sources`: the benchmark suite samples `max(50, 4000 // n_labels)` per label, which gives
about 800 calib items. Selection compares models on calib, and about 200 items cannot separate a 2-point difference.

**Segment suites need enough calib files.**
`byosm/evals/suites.py` `_fetch_segments`: frame tasks compute the calib objective per file, so few files make selection
noisy; segment suites take `max(8, max_files // 4)` calib files.

**Results are cached by fingerprint, and the model is not in it.**
`byosm/evals/suites.py` `fingerprint`, `byosm/evals/board.py` `_cached_or_scored`: the cache key covers the suite
definition, the first `FINGERPRINT_CHARS_PER_FILE` (100,000) characters of each frozen suite file, the decision
policy, the `data_key` for builtin, and `SCORE_VERSION`. Bump `SCORE_VERSION` whenever `score()` output changes. A
partial `evals.run` updates `board.json` and drops entries for experiments, baselines or suites that no longer exist.
Gotchas: the key does not cover the model weights, the base objective or, for builtin, the dataset contents. Rebuilding
an existing experiment name (`byosm build x --exp v3` after editing its `spec.yaml`), rebuilding data with
`--force-data` or `byosm data --force`, or changing `objective.metric` keeps the old scores, and `maybe_promote` then
decides on them. Nothing clears `experiments/<exp>/results/`; run `byosm eval --force` after any of these.

**Result files stay diffable.**
`byosm/evals/dumps.py` `_dumps`: indented JSON, but numeric arrays (confusion matrices, units) stay on one line.

**Baselines are the bar.**
`byosm/predictors/`: `energy` is the floor (frame log-energy through a sigmoid at -45 dB, 4 dB scale); a model that
cannot beat it is broken. `llm` is the System-2 bar: the `baseline` role, zero-shot on the label definitions; `llm:<role>`
picks another role. `hf:<repo>` runs any Hub classifier, matching labels by normalised name, dropping unmatched
probability and renormalising; discovery keeps one only if it covers at least 60% of our labels.

**The LLM baseline sees the originals.**
`byosm/predictors/zero_shot.py`: `wants_rows` gives it the original files, not preprocessed tensors. Document pages are
sent as images up to 1000 px plus the first 1500 characters of text layer or OCR. `images` is passed only when there are
images, so text-only providers need not accept it. Latency is NaN: it is network-bound and not comparable.

**Silero is the exact LiveKit model and is favoured.**
`byosm/predictors/audio.py`: `SileroVAD` uses the ONNX bundled with livekit-plugins-silero; 545,601 parameters count its
8 kHz and 16 kHz branches. Each frame maps to the 512-sample window containing its start, which gives Silero up to 32 ms of
look-ahead. Gotcha: Silero and energy assume 16 kHz but do not resample. They get audio loaded at
`spec.data.audio.sample_rate` and only compute their hop at 16 kHz (`frame_hop(spec, SAMPLE_RATE)`), so with any other
spec rate their frames are misaligned and Silero gets the wrong input rate.

**Experiments are scored the way they run.**
`byosm/predictors/experiment.py`: streaming backbones are scored by streaming the whole recording (160,000-sample spans
that carry state). Others run on independent windows of the training clip length, resampled to the spec's frame hop.

**Streaming cost is per chunk on one thread; other latency is classify only.**
`byosm/predictors/experiment.py` `stream_chunk_ms`, `byosm/evals/scoring.py` `_add_speed`: a streaming model is timed
on 1000 chunks of 30 ms on a single torch thread, as Silero runs in LiveKit; the total seconds for 1000 chunks is
numerically the milliseconds per chunk. Otherwise the board's latency is the predictor's time on the first test item,
and only for classify tasks, so a non-streaming frame model (wav2vec2) reports no latency. The only whole-input timing
is in `byosm/bench/latency.py`.

**The thesis benchmark gives the generalist the biggest budget, not the same recipe.**
`byosm/bench/`: N specialists of one modality (mixed modalities are refused) against one shared-backbone multi-task
model trained on the union of their train and dev data. The
generalist gets the largest specialist backbone by default and the most epochs and pretraining steps among them, but
its other hyperparameters come from the first spec, and it trains with plain AdamW in two flat LR groups: no layer-wise
decay, probe epochs, EMA or mixed precision. Latency is the time to answer every task for one input: N specialists in
parallel against one shared forward pass plus N heads. Gotcha: both sides are scored on each specialist's builtin test
split (`_task_row`), with no check of `n_test_folded`. A promoted specialist trained on that split (doctype `v2-ditft`
and intent `final-s1` both have `n_test_folded` set) is graded on its own training rows, which tilts the comparison
toward the specialists.

## Improve and autopilot

**A proposal is a small spec edit plus its evidence.**
`byosm/improve/`: ops `set`/`append` on dotted spec paths (lists by index) and a `why`. Built-in rules always run in
registration order, and among rule proposals the first with a given id wins (`registry.py` `rule_proposals`). Add a
rule with `@rule`. LLM proposals from the `analyze` role get an `llm-` id prefix and are not deduped. Gotcha: they are
filtered op by op (`analyst.py` `_parse`): an op outside `ALLOWED` is dropped silently and the rest of the proposal is
still applied, so a partly filtered proposal may no longer do what its `why` says.

**Allowed paths are dotted patterns, not a leakage check.**
`byosm/improve/paths.py`: `|` separates alternatives for one segment and `*` is a list index. The same text is shown to
the LLM, so it knows what it may edit. Gotcha: `ALLOWED` lets an LLM proposal append any `data.hf_sources` entry or set
its `partition`, including the calib or test of a suite's own source; nothing checks proposals against the suites.
`backbone` and `document.image_backbone` are allowed too, so an LLM proposal can pick an encoder fine-tuned on the
benchmark (see "Domain-pretrained encoders").

**Converging late.**
`byosm/improve/diagnosis.py` `_converging_late`: the best dev epoch is in the last quarter of a run of at least 4 epochs
that was not early-stopped (see "Epochs are chosen on dev"). Both `more-data` and `train-longer` use it.

**Real data from the suite's own source.**
`byosm/improve/rules_data.py` `real_data`: when an external suite lags a baseline (or the builtin suite by more than
0.05) and the model never trained on that source, the rule adds that source for training. For `segment_files` it reads
the train hash partition, which is disjoint from the suite's calib/test. Gotcha: for rows sources it only switches the
split to `train` and keeps the suite's partition, so if the suite itself reads `train` the new source overlaps it.

**More data until the split runs out.**
`byosm/improve/rules_data.py` `_more_balanced`: a balanced source asks for 4x per label, capped at what the rarest label
has. When less than 1.5x is left, it adds the dataset's largest split that no suite or source reads (for example RVL-CDIP
train while training on val). `availability()` counts rows after `label_map` and partition filtering.

**Domain-pretrained encoders that never saw the benchmark.**
`byosm/improve/rules_fit.py` `DOMAIN_ENCODERS`: when the primary suite lags the best baseline by more than 0.02, a
general encoder is swapped for one pretrained on the domain. For scans the proposal is `hf:microsoft/dit-base`,
self-supervised on unlabelled document scans, not the RVL-CDIP fine-tuned DiT: that one saw the rvl benchmark's labels
and is a baseline, so proposing it would leak. On doctype, fused (0.8009 rvl test) to fused-dit (0.8838) was the
largest jump, though that experiment also added data and epochs. Gotcha: this protects only the built-in rule. The
promoted doctype model, `v2-ditft` (0.9481 rvl test against the baseline's 0.94), uses
`hf:microsoft/dit-base-finetuned-rvlcdip`, set by hand (proposal `manual`), and `ALLOWED` would let an LLM proposal set
it as well. Its rvl result is not a clean held-out number.

**Label convention for frame audio.**
`byosm/improve/rules_fit.py` `label_convention`: if calibration pushes a suite's threshold to 0.2 or below or 0.8 or
above, that suite labels differently from training (gaps over `fill_ms` are negative in training; segment suites usually
label whole utterances). The fix is `fill_ms: 300`.

**Cost estimates.**
`byosm/improve/cost.py`: `INGEST_S` is seconds per new labelled item, measured on one machine: 0.36 s for documents (OCR,
one engine per core), media decode for the rest. Visual-only documents cost as images. A changed encoder multiplies
training time by 1.5 because its speed is unknown.

**One proposal per round.**
`byosm/auto/loop.py`: each round applies exactly one proposal, so each gain is attributable to one proposal. Gotcha: a
proposal can carry several edits (`regularize` halves the LR and raises `modality_dropout`, `hard-pairs` adds hard cases
and a refine round, an LLM proposal can carry many ops), and a manual build can apply several proposals at once
(doctype `fused-dit` applied `more-data`, `domain-encoder` and `train-longer`).

**The experiment budget counts every experiment.**
`byosm/auto/loop.py` `_budget_stop`: `max_experiments` (default 8) is compared with all experiments in the project,
manual ones included. `byosm auto` on an existing project can stop before trying anything; doctype already has 8.

**Priors, then cost, then rule order.**
`byosm/auto/loop.py` `_pick`: `PRIORS` gives the expected gain per proposal kind (by id prefix, default 1.0). Ties go to
the cheaper proposal, then to rule order. A proposal with unknown (NaN) cost counts as fitting the budget, because
`NaN > left` is false.

**Any calib gain is kept; only a real gain resets the plateau.**
`byosm/auto/loop.py` `_is_real_gain`, `byosm/pipeline.py` `maybe_promote`: an experiment is promoted (kept) on any strict
calib improvement, so a noise-level winner is kept, distilled and shipped. Significance only drives stopping: a round
counts as a real gain if P(better on calib) is at least 0.9 under the paired bootstrap, or, without one, the calib value
rises by more than 0.002. Two rounds without a real gain stop the loop; a failed build counts as stale but does not
trigger the plateau check in that round. Gotcha: the calib value of the best of many tries is biased upward (winner's
curse), so read the test column, not calib, for how good the kept model is.

**Autopilot stops for a reason and says so.**
`byosm/auto/autopilot.py`: define, build v1, loop, then stop when it has won (beats every baseline by `margin` on calib
and meets the target), hits the experiment or time budget, plateaus, or runs out of proposals. A winner above
`max_params_m` is distilled. Every decision and its evidence go to `auto.json` and `auto.md`. Gotchas: the student is
promoted when its test objective beats the best baseline's calib value plus `margin` (two splits, and the target is
ignored); and building the student already ran `maybe_promote`, so a student with a better calib value than the teacher
stays promoted even when the log says "keep the teacher".

## Export and runtime

**The bundle is self-contained.**
`byosm/export/`: `specialists/<name>/export/` (or `--out`) holds `model.onnx` (outputs `embedding` and `logits`),
`stream.onnx` for causal audio, `manifest.json` (labels, calibrated decision, frontend, metrics), the tokenizer or mel
assets, `byosm_rt.py`, `requirements.txt` and a README.

**The fp32 export is checked against PyTorch.**
`byosm/export/verify.py`: on five real test inputs the max absolute difference must be at most 1e-3, and streamed output
must match offline output too, or export fails. This runs on the fp32 `model.onnx` before quantisation; an int8 model
that ships replaces it and is judged only by the calib check below.

**int8 only when it is free.**
`byosm/export/__init__.py`, `quantize.py`: the `precision` argument of `export()` and `byosm export --precision` is
`auto`, `int8` or `fp32` (not the spec's `train.precision`). Quantisation is never tried for the streaming log-mel CRNN
or for a `model.onnx` of 4 MB or less, even with `int8`. Otherwise it tries MatMul/Gemm plus Gather first, then
MatMul/Gemm, and with `auto` ships the first variant whose primary-suite calib objective (never test) drops by at most
0.5 pt, else fp32 with `rejected` in the manifest under `quantization`. Gotcha: only classify tasks have a calib
evaluator, so frame tasks stay fp32 with `auto`, and `int8` ships the first variant with no check at all.

**Document export traces two pages.**
`byosm/export/frontends.py` `_document`: the example uses `min(2, max_pages)` pages so it fits the page-position table.
In the graph the pages of one document are the batch dimension of the page encoders.

**One runtime file, numpy and onnxruntime only.**
`byosm/rt.py`: copied into every bundle as `byosm_rt.py`. Extra packages are needed only per input type (tokenizers,
Pillow, pypdfium2/rapidocr, soundfile). It must run on Python older than 3.10, so it uses
`from __future__ import annotations` and nothing like `zip(strict=...)`. API: `load(dir)`, `encode`, `predict`,
`stream().step(chunk)`, `serve([...], port=8800)`; HTTP `GET /v1/models`, `POST /v1/<name>/predict|encode`, and
`POST /v1/predict` to broadcast to every model.

**The numpy log-mel matches torchaudio.**
`byosm/rt.py` `LogMel`: it reproduces `MelSpectrogram(center=False)` (power 2, HTK scale, no norm) with the same causal
left pad. The window and filterbank ship as `window.npy` and `mel_fb.npy` so the runtime needs no torch. `StreamSession`
keeps the last `left_pad` samples between chunks so streamed frames line up with offline frames.

**Hysteresis copies LiveKit.**
`byosm/rt.py` `HysteresisState`: EMA smoothing with alpha given per 32 ms and rescaled to the hop, deactivation at
`act - deact_offset` (floor 0.01), min on/off durations, and start/end events that fire with LiveKit's delays while region
boundaries are backdated to where the run began. Training-side scoring imports the same function.

**Handler names are fixed.**
`byosm/rt.py`: `BaseHTTPRequestHandler` dispatches by the names `do_GET` and `do_POST`, so they cannot be renamed.

## Publishing

**The model card is generated from artefacts.**
`byosm/hub/`: the card cannot drift from what was measured. `spec.yaml` gives task, labels and data; `board.json` gives
test results next to every baseline; the `proposal.json` chain gives the lineage of `improve` proposals that produced the
model; the export manifest gives size, decision policy and usage.

**README tables are generated too.**
`byosm/report/`, `byosm/cli/report.py`: `byosm report` rewrites the result tables between `<!-- report:<name> -->`
markers from `specialists/*/board.json`, and `--check` writes nothing and exits 1 when a block is stale. Same idea as
the model card: README numbers cannot drift from the results files. Rows scored on an older suite fingerprint are
marked with a dagger.

**Seed groups are a naming convention.**
`byosm/report/seeds.py`: experiments named `<recipe>-s<N>` form one recipe group, and a group of at least `MIN_SEEDS`
(2) gets a mean and sample sd in the `report:<name>:seeds` block. Seed noise is how large a gap must be before one
recipe beats another; a one-seed difference smaller than that sd means nothing.

**`model-index` front matter.**
`byosm/hub/tables.py` `front_matter`: the Hub renders it as an evaluation widget on the model page.

**One licence default, private by default.**
`byosm/config.py` `PUBLISH_LICENSE`: everything published is CC BY-NC 4.0 (`cc-by-nc-4.0`), the default for
`byosm publish`, `byosm auto` and `byosm.auto.run`. `byosm publish` creates a private repo unless `--public` is passed;
autopilot always publishes private, to be flipped public on the Hub after review.

## CLI and configuration

**LLM roles fall back to `default`.**
`byosm/config.py`, `byosm/llm/__init__.py`: roles are `spec`, `generate`, `judge`, `eval`, `analyze` and `baseline`; any
role can use any provider. `anthropic` uses the official SDK; `openai` covers any OpenAI-compatible server (Ollama, vLLM,
LM Studio, OpenRouter). Bulk generation is the expensive part, so a cheaper `generate` model is usually fine.
`register_provider` plugs in other backends.

**Reasoning effort is a per-role knob.**
`byosm/llm/openai_provider.py`, `anthropic_provider.py`, `byosm.yaml`: the workspace default is a local reasoning model
through Ollama, and `reasoning_effort: none` (passed only when set) lets it answer without thinking. The built-in
Anthropic default sends `effort: medium`. Set either per role in `byosm.yaml`. Gotcha: the keys differ (`effort` for
`anthropic`, `reasoning_effort` for `openai`), and both providers silently ignore unknown keys, so a wrong or
misspelled key has no effect and raises no error.

**Schemas are made strict for structured output.**
`byosm/llm/base.py` `_strict`: every object gets `additionalProperties: false` and every property required, recursively,
because strict modes reject anything else.

**OpenAI-compatible servers fall back to JSON mode.**
`byosm/llm/openai_provider.py`: if the server rejects `json_schema`, the provider switches to `json_object` for good and
appends the schema to the prompt.

**Refusal fallbacks only where supported.**
`byosm/llm/anthropic_provider.py`: only the model families in `_FALLBACK_PREFIXES` accept server-side refusal fallbacks;
other models use the plain messages endpoint.

**Retries are bounded.**
`byosm/llm/base.py` `_retry`: JSON parse errors and `LLMError` retry at once; other errors retry only when they look
transient (rate limit, overload, timeout, connect), with 2, 4, 8 s backoff. `pmap` reports failed items through
`on_error` and skips them.

**Commands register by import.**
`byosm/cli/__init__.py`: importing the command modules (`define`, `iterate`, `report`, `ship`, `train`) is what
registers their `@app.command`s, so a new command module must be imported there.

**Commands that just scored print the detailed board.**
`byosm/cli/train.py`, `byosm/cli/iterate.py`: `build`, `eval` and `auto` call `board(name, detail=True)`, so the CI and
P/R columns appear after a run; plain `byosm board` hides them unless `--detail` is passed.

**NaN cost is skipped by self-comparison.**
`byosm/cli/iterate.py` `_log_proposal`: `minutes == minutes` is false for NaN, so proposals with unknown cost print no
estimate.
