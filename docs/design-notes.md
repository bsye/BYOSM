# Design notes

The code has no comments. This file holds the reasons behind the design: why a choice was made, what was measured, and
the gotchas. Each entry names the module it is about. Numbers are constants in the code, results in
`specialists/*/board.json`, or measurements that are labelled as such.

## Spec

**The spec is the single source of truth.**
`byosm/spec/task.py`: `TaskSpec` is compiled from one sentence and then edited as YAML. Every experiment keeps a snapshot
of it, so an experiment can be rebuilt from its own `spec.yaml`. Scoring and promotion read the base spec: its
`decision`, `objective`, `evals`, `baselines`, labels and `positive_label` (`byosm/evals/board.py`,
`byosm/pipeline.py` `maybe_promote`, `byosm/auto/standing.py`). Gotcha: training reads the experiment's spec
(`byosm/pipeline.py` `build`). In an experiment's `spec.yaml`, `decision` and `baselines` have no effect, but
`objective` still drives epoch selection, the dev threshold and the refine target check, labels still drive the
training targets, and `evals` still decides whether `holdout_test: auto` folds the builtin test split into training.

**One directory per specialist.**
`byosm/config.py` `Project`: `specialists/<name>/` holds the base `spec.yaml`, `data/<data_key>/`, `evals/<suite>/`
(frozen suites), `baselines/<id>/` and `experiments/<exp>/` (model, snapshot spec, `meta.json`, `results/`), plus
`board.json`, `proposals.json`, `cache/`, `export/`, the autopilot's `auto.json` and `auto.md`, and a `promoted` file
naming the shipped experiment. Only directories with a `meta.json` count as experiments.

**Discovery turns a public benchmark into the primary suite.**
`byosm/discover.py` `_add_benchmark`, `_split_sources`: for classify tasks, when a matching Hub dataset is found, its
held-out split becomes a balanced `hf` suite named `bench` and the new primary (every other suite stops being
primary), and its `train` split, if there is one, is added as a balanced training source. Default baselines: `llm` for
text, document and image classification; `energy` and `silero` for frame-level audio; plus Hub classifiers found by
search. Gotcha: a primary suite that is not builtin turns on `holdout_test: auto` folding (see below), so discovery
also decides that the builtin test split goes into training.

**`data_key` decides when data is shared.**
`byosm/spec/task.py` `data_key`, `SYNTH_FIELDS`: the key hashes the modality, task type, label names, frame hop and the
`data` section without its default values, so a new `data:` field with a default leaves every key unchanged. Experiments with equal keys reuse the same `data/<key>/`. With `data.synthetic` on it also
hashes what the LLM reads to write data: label definitions, `description`, `name`, `summary`, `input_description`,
`languages`, `diversity_axes` and `hard_cases`. Editing those on a real-data spec keeps the cached data, because no
generator reads them. Gotcha: refine rounds append to the shared `train.jsonl`, so a later experiment with the same
key inherits them.

**The backbone catalog is a menu, not a limit.**
`byosm/spec/backbones.py`: `BACKBONES` is what the compiler may choose from; any `hf:<repo>` works when set by hand. An
unknown id from the LLM falls back to `DEFAULT_BACKBONE`: the smallest entry per modality, except text, which gets
MiniLM-L6 instead of tiny BERT.

**Hash partitions keep segment files apart.**
`byosm/data/segments.py`: `partition_of` puts each item in train (70%), calib (6%) or test (24%) by an md5 of its key.
`segment_files` suites read the calib and test partitions of the file names. A `segment_files` training source with
`partition: train` or `all` reads train and does not overlap them; any other partition is read as given, so `calib`
or `test` overlaps the suite (`byosm/data/__init__.py` `_segment_clips`). Balanced parquet sampling also filters rows
by the md5 partition of `<file>:<row>`, but only when the source sets a `partition` other than `all`
(`parquet.py` `_in_partition`); `sampling: stream` ignores `partition`. Gotcha: discovery leaves `partition` at `all`,
so its `bench` suite and its training source are kept apart only because they read different splits.

**Balanced sampling exists for label-sorted datasets.**
`byosm/data/hf_source/parquet.py`: RVL-CDIP and similar datasets are sorted by label, so a stream, even through its
1000-row shuffle buffer (`rows.py`), only ever sees the first labels. `balanced_rows` reads only the label column of the
split's parquet files to build an index, draws `per_label` random rows per label (seed 0), then decodes only those
rows. Gotcha: every parquet file of the split is downloaded to build the index, and the whole input column of each
file holding a chosen row is read, so for image datasets such as RVL-CDIP most of the bytes are fetched anyway. Files
are cached by `hf_hub_download`.

**`label_map` filters, and its empty form differs per reader.**
`byosm/data/hf_source/rows.py`, `parquet.py`, `byosm/data/__init__.py`: pairs are `[source_value, our_label]` and
unlisted source values are dropped. With an empty map, balanced parquet sampling matches source labels to ours by
normalised name, while stream sampling gives every row the label `None`, so labelled rows are lost. To filter rows
for positive/negative/unlabeled sources, rows sources map wanted values to `keep` (`[value, "keep"]`), but
`segment_files` sources use the reverse order (`["keep", cluster]`). Gotcha: `segment_files` eval suites ignore that
filter (`byosm/evals/suites.py` `_fetch_segments`) and mark every annotated segment positive, so training and suite
labels can disagree with no warning.

**Audio frame labels follow the signal.**
`byosm/spec/sections.py` `AudioCfg`, `byosm/data/audio_mix.py` `frame_activity`: with `label_by_energy`, frames come
from the positive clip's own energy, so pauses inside speech are negative. A frame is active within `rel_db` (35 dB)
of the event's peak and above -55 dB. Gaps of up to and including `fill_ms` stay positive: about 60 ms gives
word-level labels, about 300 ms utterance-level labels.

**Only binary tasks get a fitted threshold.**
`byosm/spec/sections.py` `Decision`, `byosm/metrics.py` `_binary`, `calibrate`: `threshold` is the only decision value
that can be left empty. A `None` threshold is fitted on each suite's calib split only for binary tasks: `positive_label`
set, or two labels with objective `f1`, `recall_at_precision` or `precision_at_recall`. Every other task (multi-class,
or two labels with `macro_f1`/`accuracy` and no `positive_label`) decides by argmax and calibrates nothing. The fit
maximises the objective for `recall_at_precision` and `precision_at_recall`, binary F1 otherwise. For hysteresis only
the activation is fitted (see "Hysteresis activation is fitted for F1"); the other values are used as written and
follow LiveKit (see "Hysteresis copies LiveKit").

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
the separation (see "LLM roles fall back to `default`" for how a role entry is merged).

**A second labeller removes label noise.**
`byosm/data/synth_text.py` `judge`: the `judge` role relabels each generated row blind, and only rows where both labels
agree are kept. Label noise caps the precision a tiny model can reach, and this is cheap insurance. It is a second
opinion only when `judge` is configured as a different model.

**Refinement learns from dev errors.**
`byosm/pipeline.py` `refine`, `byosm/data/synth_text.py` `refine_jobs`: the `analyze` role reads the dev errors and
returns failure patterns with data recipes; the `generate` role then writes targeted training rows from them (filtered
by `judge` when it is on), deduped against train, dev and test. The refine prompts come from the patterns and do not
read `hard_cases`. The refine loop itself never sees test rows, but dev is the builtin suite's calib split (see "The
builtin calib split is also the training dev split"). Gotcha: each round retrains from scratch on the grown
`train.jsonl` and overwrites `model.pt` and `meta.json` (`byosm/train/outputs.py` `save_run`). The experiment keeps the
last round's model, not the round with the best dev score, so a round that makes dev worse still replaces a better
model.

**Text held-out sets are deduped against train.**
`byosm/data/__init__.py`, `byosm/data/store.py` `dedupe`: for text, test is deduped against train+dev and dev against
train. A row is dropped if it is an exact normalised duplicate of a reference row, or if its 5-gram Jaccard is above
0.7 with one of the 3 reference rows that share the most shingles with it. Rows within one split are deduped only
exactly. Gotchas: no other modality is deduped. The builtin suite always reads the base spec's data, so an experiment
with a different `data_key` (for example after `more-data`) re-splits its own rows and can train on base dev/test rows.

**Procedural audio has exact labels.**
`byosm/data/audio_mix.py`: clean positive events are placed over negative backgrounds at controlled SNRs, so labels are
exact by construction. Pools are split into train/dev/test before mixing, so test clips never reuse train source audio.
About 12% of backgrounds (all of them when there are no negatives) are white noise instead of negative audio. Every
background is then rescaled to an RMS between -45 and -20 dB (`Mixer.clip`), so no near-silent background reaches
training.

**Model selection also sees real audio, though not independent of train.**
`byosm/data/__init__.py` `_collect`: one in ten real segment-labelled clips (for example AVA-Speech) goes to dev instead
of train, so dev is not only procedural mixes. Gotcha: those clips are adjacent windows of the same recordings as the
train clips.

**An empty dataset is never cached.**
`byosm/data/__init__.py` `build`: it raises before writing any split when there are no training examples, so a bad data
plan does not leave a `train.jsonl` that later runs would reuse.

**Media is decoded without ffmpeg.**
`byosm/data/hf_source/decode.py`, `hub.py`: audio and images are decoded with soundfile and PIL, and streaming datasets
get their Audio/Image columns cast to `decode=False`, so the `datasets` decoder (and ffmpeg) is never needed.

**ClassLabel names come from the parquet schema first.**
`byosm/data/hf_source/parquet.py` `_class_names`: the `huggingface` metadata in the parquet schema is read when present,
which avoids loading the dataset builder; the builder info is loaded only as a fallback.

**When no dataset fits, ask for one instead of failing.**
`byosm/data/request.py`, `check.py`, `probe.py`: the autopilot stops before the first build when some labels have no
labeled source and the input cannot be generated (anything but text with `synthetic`), and writes
`data_request/`: `template.csv`, `template.xlsx` (label and split dropdowns, a `labels` sheet with definitions and
borderline tips), `request.md` and `request.json`. Test rows are sized so accuracy is measured to within 3 points at
95%: `1.96² p(1-p) / 0.03²` with `p` the target capped at 0.9, because a near-perfect target would otherwise ask for
almost nothing. Training rows start at 50 per label (200 recommended) and double for labels the LLM calls confusable;
the learning curve in `byosm improve` refines that after the first build. `check-data` blocks on rows that would
corrupt training or the score (unknown labels, missing files, empty inputs, the same input with two labels, a test row
duplicated in training) and warns on shortfalls. Likely mislabels come from a cross-validated logistic probe on MiniLM
embeddings of the text (or of the document's text, OCR'd and cached): only rows where the probe is confident and
disagrees are flagged, and the judge LLM reviews at most 50 of them. A usable table becomes `data.local_dir`; a table
path there reads rows, a folder keeps the one-sub-folder-per-label layout, so no new `data:` field (and no data key
change) was needed. Filled tables are data and are git-ignored.

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

**Page, OCR and pooling settings.**
`byosm/spec/sections.py` `DocumentCfg`: `max_pages` keeps the first pages. `ocr: auto` runs OCR only on pages whose text
layer has fewer than 20 characters (scans). `image_pool: pooler` uses the encoder's own trained pooling (for BEiT/DiT,
patch mean plus LayerNorm), which is what its fine-tuned head read.

**Only PDF rendering is serialised.**
`byosm/modalities/document/pages.py`, `byosm/rt.py`: PDFium is not thread-safe but RapidOCR is, so a lock covers
rendering only. Training renders in threads (`Document.load_many`), then OCRs every page missing from the store in one
batch; the runtime needs the lock because its `ThreadingHTTPServer` and broadcast thread pool render concurrently.

**One single-threaded OCR engine per core.**
`byosm/modalities/document/engine.py` `ocr_many`: the OCR models are small, so intra-op threads mostly synchronise; one
engine per process with one thread each was measured about 2.3x faster than one engine on every core. Batches smaller
than `2 x cpu_count` stay in-process because pool start-up would dominate. A new pool is made per batch: start-up takes
seconds, while idle workers would hold several GB of RAM for the whole training run.

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
`byosm/train/common.py` `param_groups`, `_encoder_groups`: low layers hold general features worth keeping, so the LR
falls geometrically from `lr_backbone` for parameters with no layer index (pooler, final norm) to
`lr_backbone * layer_decay_floor` for the embeddings; the highest numbered layer gets
`lr_backbone * layer_decay_floor ** (1 / top)`. Layers trained from scratch (heads, projections, fusion) get `lr_head`.
A backbone with no pretrained encoder (tiny-crnn, tiny-cnn) trains at `lr_backbone`. Depth is read from parameter
names: a `layer|layers|blocks|block|h|stages` segment followed by an integer; names with `embed` are depth 0.

**Mixed precision only for pretrained encoders.**
`byosm/train/common.py` `amp_dtype`: `train.precision: auto` uses fp16 on MPS and bf16 on CUDA (when supported) only
when the model has a pretrained encoder, where it was measured 2.1x faster per step (fp16, MPS, DiT+MiniLM); small
from-scratch recurrent nets stay in fp32.

**EMA over about half an epoch.**
`byosm/train/trainer.py` `_ema`: decay is `1 - 1/max(10, steps_per_epoch/2)`, which smooths SGD noise without lagging
behind the cosine LR schedule. The EMA weights are the ones evaluated and shipped.

**Probe epochs are capped and frozen encoders are in eval mode.**
`byosm/train/trainer.py`: linear-probe epochs are `min(probe_epochs, epochs // 4)` and are skipped with
`freeze_backbone` or without a pretrained encoder. A frozen encoder is switched to `eval()` so it is a fixed feature
extractor: no dropout, no BatchNorm statistic updates.

**A missing dev split is cut from train.**
`byosm/train/loop.py` `_split_rows`: a random 10% of train (at least one row) is held out with the run's seeded RNG, so
epoch selection always has a dev set. Gotcha: folded builtin test rows are added to train first, so this dev can
contain them.

**Epochs are chosen on dev, with early stopping.**
`byosm/train/loop.py` `_select`, `PATIENCE`: after each epoch the dev objective is measured and the best epoch's weights
are snapshotted; ties go to the higher dev macro F1. Training stops after `PATIENCE` (4) epochs without a new best, and
the snapshot is what is saved. Gotcha: the dev threshold is fitted on dev itself (`_dev_epoch` passes `calibrate=True`
to `byosm/train/evaluate.py`), so `meta["dev"]` is optimistic in the way cross-fitted calib scores avoid. Epoch
selection, early stopping and the refine loop's target check in `byosm/pipeline.py` `build` all use that value.

**Frame tasks are selected with a different decision rule than they ship with.**
`byosm/train/evaluate.py` `evaluate_rows`, `byosm/evals/scoring.py` `score`: training decides frames with a plain
per-frame threshold fitted on dev, and that threshold is saved as `meta["threshold"]`. The board, promotion and the
export decide with hysteresis and an activation fitted on calib. So a VAD model's epoch is picked under one rule and it
is scored and shipped under another. `byosm bench` uses the per-frame rule on both sides.

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

**A human review beats the panel.**
`byosm/data/review.py`, `byosm/data/panel.py`: `specialists/NAME/reviewed.jsonl` holds one label per input (the latest
wins). Reviewed inputs are kept first when the pool is capped at `teacher_rows`, still go through the teachers so the
review table can show their votes, and then train on the human label as a one-hot target with source `reviewed`. The
file sits next to the spec, not in the data folder, so a review survives a data rebuild; it is read only when the data is
built, which is why a review needs `byosm data NAME --force` (or a new data key) to take effect.

**TinyCRNN is causal and streams exactly.**
`byosm/modalities/audio.py` `TinyCRNN`: the waveform is left-padded by one window minus one hop, so mel frame `t` ends at
sample `(t+1)*hop` and lines up with the frame labels. `stream()` carries the waveform tail, per-conv left context and
GRU state, and matches `forward()` on the concatenated audio.

**Frame targets follow the model's frame rate.**
`byosm/model.py` `align_targets`: when label rate and backbone frame rate differ (10 ms labels, 20 ms wav2vec2 frames),
targets are nearest-resampled to the output length. Padding uses `IGNORE`.

**Word-level text labels go on the first sub-token.**
`byosm/modalities/text.py` `collate`: later sub-tokens and special tokens stay `IGNORE`. Gotcha: the runtime
(`byosm/rt.py` `_text_inputs`, `predict`) tokenizes the raw string, not split words, and returns one label per token,
special tokens and later sub-tokens included, so its frames do not line up with words.

**Images use the encoder's own normalisation.**
`byosm/modalities/image.py` `norm_for`: mean/std come from the encoder's `AutoImageProcessor`; ImageNet stats are the
fallback for tiny-cnn or when the processor has none.

## Evaluation and selection

**One scorer for everything.**
`byosm/evals/scoring.py`, `byosm/predictors/base.py`: every predictor, ours or a baseline, returns probabilities over the
spec's labels and is scored by the same code. Calib probabilities fit the decision policy; test probabilities give the
objective, P/R/F1 and, for binary tasks with both classes present, ROC-AUC and AP.

**Tagging is a frame task over words.**
`byosm/rt.py` `spans`, `first_subtokens`, `byosm/metrics.py` `span_counts`: `frame` means one label per unit, a 10 ms
audio frame or a word, so tagging reuses the frame path instead of adding a parallel one. A word's prediction is read
from its first subtoken, in training, scoring and the bundle alike. Spans are decoded from `O`, `B-X` and `I-X` tags (a
bare `X` counts as `I-X`) by one function in the runtime that the scorer imports. `span_f1` is entity-level F1, the
CoNLL standard: a span counts only if its type and both ends are right. Each sentence stores its (hit, extra, missed)
span counts as its bootstrap unit, so CIs and paired comparisons work as for every other task.

**Every predictor answers inside the benchmark's label space.**
`EvalSuite.label_space`, `byosm/evals/scoring.py` `_within`, `byosm/predictors/zero_shot.py`: a spec may carry labels
a public benchmark never uses (the autopilot adds a catch-all `other`). The suite's declared `label_map` targets are
that benchmark's label space; the LLM is offered only those labels and every other predictor has the rest masked before
the argmax. It comes from the suite definition, never from the gold labels. Without it, Qwen 3.8 put 37% of
dair-ai/emotion test items in `other` and scored 0.40. A narrowed space is part of the suite fingerprint.

**Every experiment is scored with the base decision policy.**
`byosm/evals/board.py` `_experiment_predictor`: the experiment's own `decision` is replaced by the base spec's, so all
experiments and baselines are compared under one policy. `decision.*` is not an improve path.

**Promotion decides on calib, with gaps.**
`byosm/evals/selection.py` `selection_value`, `byosm/pipeline.py` `maybe_promote`, `byosm/auto/standing.py`: promotion,
autopilot keep/revert and the win check compare the primary suite's calib objective. Test is reported with a 95% CI and
a paired bootstrap against the best baseline. Gotchas:
- With no baseline result on the primary suite, `standing` counts "beats every baseline" as true. A baseline that fails
  to score is dropped from `board.json`, so autopilot can then win, distill and publish without beating anything. The
  intent spec lists `llm`, but its `board.json` has no `llm` row today.
- `selection_value` falls back to the test objective when a result has no `calib_objective`: a `local` suite with no
  `split: calib` rows, an empty calib split, or an old cached result. Selection then decides on test.
- The improve diagnosis (`byosm/improve/diagnosis.py`, `registry.py` `gap`) reads the test objective, so the lag that
  triggers `real-data`, `more-data` and `domain-encoder`, and the weakest labels and confusions shown to the LLM
  analyst, all come from test. Test steers which experiments are built.
- The `hard-pairs` rule (`byosm/improve/rules_fit.py` `confusable_pairs`) copies the top test-split confusions into
  `hard_cases`. On a synthetic spec that changes the `data_key`, so the experiment builds fresh data and generation
  weaves the hard cases into train (up to 2 per job) and into its own dev and test (up to 4 per adversarial job). Test
  therefore shapes training data. The board's builtin suite still reads the base spec's data.

**The builtin calib split is also the training dev split.**
`byosm/evals/suites.py` `materialize`, `byosm/train/loop.py`: when the primary suite is builtin, its calib split is the
base data's `dev` split. Every experiment with the base `data_key` already used that split for epoch selection, early
stopping, its dev threshold and refine. Promotion on builtin therefore compares values tuned on the selection split;
an external primary suite avoids this.

**Calib decisions are cross-fitted.**
`byosm/evals/scoring.py` `_cross_fit_frames`, `_cross_fit_rows`: deciding calib with a threshold fitted on calib would
bias the selection value upward. For hysteresis, even files are decided with the activation fitted on odd files and
the other way round; for thresholds the same is done on even/odd rows (binary tasks with an empty threshold). Gotcha:
with a single calib file the frame decision is not cross-fitted.

**Hysteresis activation is fitted for F1.**
`byosm/evals/scoring.py` `_fit_activation`: when `decision.threshold` is empty, the activation is picked from a grid of
33 values in [0.1, 0.9] by frame F1 on calib, whatever the spec's objective.

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
only when marked `split: calib`. `hf` rows suites, stream or balanced, shuffle the rows they take and send every fifth
item to calib; the hash partition never splits them. Gotcha: editing a frozen suite's definition (`per_label`,
`source`, `max_items`) changes its fingerprint, so everything is rescored, but on the old frozen rows. Delete
`evals/<suite>/` to fetch the new definition.

**About 800 calib items for a benchmark.**
`byosm/discover.py` `_split_sources`: the benchmark suite samples `max(50, 4000 // n_labels)` per label, which gives
about 800 calib items. Selection compares models on calib, and about 200 items cannot separate a 2-point difference.

**Segment suites need enough calib files.**
`byosm/evals/suites.py` `_fetch_segments`: frame tasks compute the calib objective per file, so few files make selection
noisy; segment suites take `max(8, max_files // 4)` calib files.

**Jev is scored like any rival.**
`byosm/predictors/jev.py`: one choice question per item, the spec's label definitions as the criteria and its
description as the instructions; the returned probabilities are kept whole. The key comes from `JEV_API_KEY` in the
environment or the workspace `.env` (`byosm/config.py` `secret`, `.env` is gitignored). Identity is the endpoint and
`jev-latest`, so a new Jev version behind the same name is not detected; rescore with `--force` after a release.
Latency is measured through the API, so it includes the network.

**Results are cached by fingerprint, and our weights are not in it.**
`byosm/evals/suites.py` `fingerprint`, `byosm/evals/board.py` `_cached_or_scored`, `byosm/evals/identity.py`: the cache
key covers the suite definition (without defaults, like the decision policy), the first `FINGERPRINT_CHARS_PER_FILE` (100,000) characters of each frozen suite file,
the decision policy, the `data_key` for builtin, and `SCORE_VERSION`. Bump `SCORE_VERSION` whenever `score()` output
changes. `llm` and `hf:` baselines also store an `identity`: the role's resolved LLM config without `api_key`, or the
Hub repo plus the revision in the local Hugging Face cache. A changed identity rescores that baseline, so does any
change to the role's resolved config, even a key the provider ignores (both accept unknown keys, such as `timeout`,
and drop them). A cache written before identities existed is accepted once and gets the
current identity. When the identity cannot be computed (an API key not set), the cached result is reused with a log
line. When scoring fails, the old entry is dropped from `board.json`, so a stale score never stays on the board. A
partial `evals.run` updates `board.json` and drops entries for experiments, baselines or suites that no longer exist.
Gotchas: the key does not cover experiment weights, the base objective or, for builtin, the dataset contents.
Rebuilding an existing experiment name (`byosm build x --exp v3` after editing its `spec.yaml`), rebuilding data with
`--force-data` or `byosm data --force`, or changing `objective.metric` keeps the old scores, and `maybe_promote` then
decides on them. Nothing clears `experiments/<exp>/results/`; run `byosm eval --force` after any of these.

**Result files stay diffable.**
`byosm/evals/dumps.py` `_dumps`: indented JSON, but numeric arrays (confusion matrices, units) stay on one line, so a
rescore shows up as a readable diff.

**Baselines are the bar.**
`byosm/predictors/`: `energy` is the floor (frame log-energy through a sigmoid at -45 dB, 4 dB scale); a model that
cannot beat it is broken. `llm` is the System-2 bar: the `baseline` role, zero-shot on the label definitions; `llm:<role>`
picks another role. `hf:<repo>` runs any Hub classifier, matching labels by normalised name, dropping unmatched
probability and renormalising; discovery keeps one only if it covers at least 60% of our labels. Gotcha: image and
document `hf:` baselines see only the first page. Add a baseline with `@register_baseline(name, summary)`
(`byosm/predictors/base.py`); it is then usable as `baselines: [name]`.

**The LLM baseline sees the originals and votes one label.**
`byosm/predictors/zero_shot.py`: `wants_rows` gives it the original files, not preprocessed tensors. Document pages are
sent as images up to 1000 px plus the first 1500 characters of text layer or OCR. `images` is passed only when there are
images, so text-only providers need not accept it. Gotchas: its probabilities are one-hot, so on binary tasks threshold
calibration and ROC-AUC/AP are degenerate for it. A label the LLM leaves out becomes a vote for the first label. It is
never timed (`_add_speed` skips `wants_rows` predictors), so its board latency is empty: it is network-bound and not
comparable.

**Silero is the exact LiveKit model and is favoured.**
`byosm/predictors/audio.py`: `SileroVAD` uses the ONNX bundled with livekit-plugins-silero; 545,601 parameters count its
8 kHz and 16 kHz branches. Each frame maps to the 512-sample window containing its start, which gives Silero up to 32 ms of
look-ahead. Gotcha: Silero and energy assume 16 kHz but do not resample. They get audio loaded at
`spec.data.audio.sample_rate` and only compute their hop at 16 kHz (`frame_hop(spec, SAMPLE_RATE)`), so with any other
spec rate their frames are misaligned and Silero gets the wrong input rate.

**Experiments are scored the way they run.**
`byosm/predictors/experiment.py`: streaming backbones are scored by streaming the whole recording (160,000-sample spans
that carry state). Others run on independent windows of the training clip length, resampled to the spec's frame hop.
Gotcha: both paths assume audio. `_windowed_frames` slices the raw input by `audio.sample_rate * clip_seconds` and the
frame hop for any frame task, so text and document frame tasks are not scored meaningfully.

**Streaming cost is per chunk on one thread; other latency is classify only.**
`byosm/predictors/experiment.py` `stream_chunk_ms`, `byosm/predictors/audio.py`, `byosm/evals/scoring.py` `_add_speed`:
a streaming model is timed on 1000 chunks of 30 ms (480 samples) on a single torch thread; the total seconds for 1000
chunks is numerically the milliseconds per chunk. Silero is timed the same way but on its own 32 ms (512-sample) chunks
through its ONNX session, so the streaming column compares different chunk sizes and runtimes. Otherwise the board's
latency is the median of 20 runs (after 3 warm-ups) on the first test item, and only for classify tasks, so a
non-streaming frame model (wav2vec2) reports no latency. `byosm/bench/latency.py` times answering every task for one
input.

**The thesis benchmark gives the generalist the biggest budget, not the same recipe.**
`byosm/bench/`: N promoted specialists of one modality (mixed modalities are refused) against one shared-backbone
multi-task model trained on the union of their train splits; their dev splits pick its best epoch (mean dev objective)
and fit its per-task thresholds. The generalist gets the largest specialist backbone by default and the most epochs and
pretraining steps among them, but its other hyperparameters come from the first spec, and it trains with plain AdamW in
two flat LR groups: no layer-wise decay, probe epochs, EMA, mixed precision, class weights or early stopping. Latency is
the time to answer every task for one input: N specialists in parallel against one shared forward pass plus N heads.
The two are not timed alike: each specialist's time includes collating and tokenising inside its `predict`, while the
generalist's shared forward is timed after collation, which flatters the generalist slightly.
Gotcha: both sides are scored on the test split of each specialist's own data (`_task_row`), with no check of
`n_test_folded`. A promoted specialist trained on that split (doctype `v2-ditft` and intent `final-s1` both have
`n_test_folded` set) is graded on its own training rows, while the generalist never trained on them, which tilts the
comparison toward the specialists.

## Improve and autopilot

**A proposal is a small spec edit plus its evidence.**
`byosm/improve/`: ops `set`/`append` on dotted spec paths (lists by index) and a `why`. Built-in rules always run in
registration order; LLM proposals from the `analyze` role follow with an `llm-` id prefix. All proposals are deduped by
id and the first wins (`registry.py` `unique`), so `improve.apply`, which selects by id, always finds exactly one. Add a
rule with `@rule`. Gotcha: LLM proposals are filtered op by op (`analyst.py` `_parse`): an op outside `ALLOWED` is
dropped silently and the rest of the proposal is still applied, so a partly filtered proposal may no longer do what
its `why` says.

**Allowed paths are dotted patterns, not a leakage check.**
`byosm/improve/paths.py`: `|` separates alternatives for one segment and `*` is a list index. The same text is shown to
the LLM, so it knows what it may edit. Gotcha: `ALLOWED` lets an LLM proposal append any `data.hf_sources` entry or set
its `partition`, including the calib or test of a suite's own source; nothing checks proposals against the suites.
`backbone` and `document.image_backbone` are allowed too, so an LLM proposal can pick an encoder fine-tuned on the
benchmark (see "Domain-pretrained encoders").

**Converging late.**
`byosm/improve/diagnosis.py` `_converging_late`: the best dev epoch is in the last quarter of a run of at least 4 epochs
that was not early-stopped (see "Epochs are chosen on dev"). It triggers `train-longer`; `more-data` only cites it in
its `why`.

**Real data from the suite's own source.**
`byosm/improve/rules_data.py` `real_data`: when an external `hf` suite lags a baseline (or the builtin suite by more than
0.05) and the model never trained on that source, the rule adds that source for training. For `segment_files` it reads
the train hash partition, which is disjoint from the suite's calib/test. Gotcha: for rows sources it only switches the
split to `train` and keeps the suite's partition, sampling and `per_label`. If the suite itself reads `train`, the new
source draws exactly the suite's rows (balanced sampling is seeded with 0), although the `why` text promises a disjoint
train partition.

**More data until the split runs out.**
`byosm/improve/rules_data.py` `_more_balanced`: a balanced source asks for 4x per label, capped at what the rarest label
has. When less than 1.5x is left, `new-split` adds the dataset's largest split that no suite or source reads (for
example RVL-CDIP train while training on val). `availability()` counts rows after `label_map` and partition filtering.
It grows the labeled source that supplied the most training rows (a source that failed to load supplied none), and
fires for any gap to the best baseline: on topic, 1,000 -> 4,000 -> all AG News rows per label took test from 0.903 to
0.921 to 0.947, the last two steps each under 0.02 behind or ahead of the best baseline.

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
`byosm/improve/rules_fit.py` `label_convention`: if calibration pushes an `hf` suite's threshold to 0.2 or below or 0.8
or above, that suite labels differently from training (gaps over `fill_ms` are negative in training; segment suites
usually label whole utterances). The fix is `fill_ms: 300`.

**Cost estimates.**
`byosm/improve/cost.py`: `INGEST_S` is seconds per new labelled item, measured on an M5 Max: 0.36 s for documents (OCR,
one engine per core; a single engine takes about 0.55 s per page), media decode for the rest. Visual-only documents
cost as images. A changed encoder multiplies training time by 1.5 because its speed is unknown. A proposal whose ops do
not apply gets a NaN cost. Gotcha: text ingest is 0, so more synthetic rows or refine rounds ignore LLM generation
time, which is the dominant cost there.

**One proposal per round.**
`byosm/auto/loop.py`: each round applies exactly one proposal, so each gain is attributable to one proposal. Gotcha: a
proposal can carry several edits (`regularize` halves the LR and raises `modality_dropout`, `hard-pairs` adds hard cases
and a refine round, an LLM proposal can carry many ops), and a manual build can apply several proposals at once
(doctype `fused-dit` applied `more-data`, `domain-encoder` and `train-longer`).

**The experiment budget counts every experiment.**
`byosm/auto/loop.py` `_budget_stop`: `max_experiments` (default 8) is compared with all experiments in the project,
manual ones included. `byosm auto` on an existing project can stop before trying anything; doctype already has 8.

**Priors, then cost, then rule order.**
`byosm/auto/loop.py` `_applicable`, `_pick`: proposals with a NaN cost (ops that do not apply to the promoted spec) are
logged as `skip` and never tried. Of the rest, those that fit the time left compete: `PRIORS` gives the expected gain
per proposal kind (by id prefix, default 1.0), ties go to the cheaper proposal, then to rule order.

**Any calib gain is kept; only a real gain resets the plateau.**
`byosm/auto/loop.py` `_is_real_gain`, `byosm/pipeline.py` `maybe_promote`: an experiment is promoted (kept) on any strict
calib improvement, so a noise-level winner is kept and exported. Significance only drives stopping: a round counts as a
real gain if P(better on calib) is at least 0.9 under the paired bootstrap, or, without one, the calib value rises by
more than 0.002. Two rounds without a real gain stop the loop; a failed build counts as stale but does not trigger the
plateau check in that round. Gotcha: the calib value of the best of many tries is biased upward (winner's curse), so
read the test column, not calib, for how good the kept model is.

**Autopilot stops for a reason and says so.**
`byosm/auto/autopilot.py`: define, build v1, loop, then stop when it has won (`standing`: beats every baseline by
`margin` on calib and meets the target), hits the experiment or time budget, plateaus, or runs out of proposals.
`_finish` distills only a winner above `max_params_m`. The student is promoted and `standing` is asked again; if the
student does not win, the teacher is promoted back, so `standing` is the only win rule. A winning student stays
promoted even when its calib value is below the teacher's. The promoted model is always exported, but published only
when it won; otherwise the log says "local only, it did not win". Every decision and its evidence go to `auto.json` and
`auto.md`. A student that fails to build is logged and the teacher stays promoted, so export and the report still
happen. Gotcha: a primary suite with no baseline result counts as beaten (see "Promotion decides on calib").

## Export and runtime

**One calibrated threshold for the bundle and the local runtime.**
`byosm/evals/selection.py` `calibrated_threshold`: the decision threshold is the one calibrated on the primary suite's
calib split (from `board.json`), falling back to the one fitted on dev during training when the board has no result.
Export writes it into the manifest and `byosm predict` / `byosm serve` use it, so a classify decision is the same on
the board, in the bundle and in the local runtime. Gotcha: for frame tasks the local PyTorch runtime
(`byosm/runtime.py`) applies that threshold per frame, while the board and the bundle apply LiveKit-style hysteresis;
use the exported bundle for streaming decisions.

**The bundle is self-contained.**
`byosm/export/`: `specialists/<name>/export/` (or `--out`) holds `model.onnx` (outputs `embedding` and `logits`),
`stream.onnx` for causal audio, `manifest.json` (labels, calibrated decision, frontend, metrics), the tokenizer or mel
assets, `byosm_rt.py`, `requirements.txt` and a README.

**Export never wipes a directory it did not write, and a failed export does not look exported to `ls` or `predict`.**
`byosm/export/bundle.py` `check_out_dir`, `clear_out_dir`, `seal`, `is_exported`: the output directory is cleared only
when it is empty or already a byosm bundle (a `.byosm-bundle` marker, or a manifest with format `byosm/1`); anything
else raises `FileExistsError`, which `byosm export` reports as a bad `--out`. The marker stays while the bundle is
written and checked. If the parity check or quantisation fails, even on Ctrl-C, `manifest.json` is deleted; the marker
is removed only on success. `byosm ls` and `byosm predict` count a bundle as exported only with a manifest and no
marker, so a killed export does not count there. Gotchas:
- The previous bundle is deleted before the new one is built and checked (`byosm/export/__init__.py` `export`), so a
  failed re-export leaves no usable bundle, not the old one.
- `byosm publish` (and autopilot's publish) only checks that `manifest.json` exists (`byosm/hub/upload.py`), and
  `byosm serve` checks nothing. An export killed during its checks keeps its manifest and marker, so it can be served
  or published unverified, and the marker is copied into the Hub stage.

**The fp32 export is checked against PyTorch.**
`byosm/export/verify.py`: on five real test inputs the max absolute difference must be at most 1e-3, and streamed output
must match offline output too, or export fails. This runs on the fp32 `model.onnx` before quantisation; an int8 model
that ships replaces it and is judged only by the calib check below.

**int8 only when it is free.**
`byosm/export/__init__.py`, `quantize.py`: the `precision` argument of `export()` and `byosm export --precision` is
`auto`, `int8` or `fp32` (not the spec's `train.precision`). Quantisation is never tried for the streaming log-mel CRNN
or for a `model.onnx` of 4 MB or less, even with `int8`. Otherwise it tries MatMul/Gemm plus Gather first, then
MatMul/Gemm, and with `auto` ships the first variant whose objective on up to 400 primary-suite calib rows (never test)
drops by at most 0.5 pt, else fp32 with `rejected` in the manifest under `quantization`. For transformers int8 is about
4x smaller (local intent export: 90.5 to 23.0 MB, calib 0.9600 to 0.9625). Gotcha: only classify tasks have a calib
evaluator, so frame tasks stay fp32 with `auto`, and `int8` ships the first variant with no check at all.

**Document export traces two pages.**
`byosm/export/frontends.py` `_document`: the example uses `min(2, max_pages)` pages so it fits the page-position table.
In the graph the pages of one document are the batch dimension of the page encoders.

**One runtime file, numpy and onnxruntime only.**
`byosm/rt.py`: copied into every bundle as `byosm_rt.py`. Extra packages are needed only per input type (tokenizers,
Pillow, pypdfium2/rapidocr, soundfile). It must run on Python older than 3.10, so it uses
`from __future__ import annotations` and nothing like `zip(strict=...)`. API: `load(dir)`, `encode`, `predict`,
`stream().step(chunk)`, `serve([...], port=8800)`; HTTP `GET /v1/models`, `POST /v1/<name>/predict|encode`, and
`POST /v1/predict` to broadcast to every model. HTTP bodies carry `input`, or `pdf_b64` for documents and `pcm16_b64`
for audio. The in-repo torch runtime for unexported models is `byosm/runtime.py` (`Specialist`, `Pool`).

**The numpy log-mel matches torchaudio.**
`byosm/rt.py` `LogMel`: it reproduces `MelSpectrogram(center=False)` (power 2, HTK scale, no norm) with the same causal
left pad. The window and filterbank ship as `window.npy` and `mel_fb.npy` so the runtime needs no torch. `StreamSession`
keeps the last `left_pad` samples between chunks so streamed frames line up with offline frames.

**Hysteresis copies LiveKit.**
`byosm/rt.py` `HysteresisState`: EMA smoothing with alpha given per 32 ms and rescaled to the hop, activation at the
threshold, deactivation at `act - deact_offset` (floor 0.01), `min_on_ms` to start and `min_off_ms` of silence to end,
and start/end events that fire with LiveKit's delays while region boundaries are backdated to where the run began.
Training-side scoring imports the same function.

## Publishing

**The model card is generated from artefacts.**
`byosm/hub/`: the card cannot drift from what was measured. `spec.yaml` gives task, labels and data; `board.json` gives
test results next to every baseline; the `proposal.json` chain gives the lineage of `improve` proposals that produced the
model; the export manifest gives size, decision policy and usage. `byosm/hub/tables.py` `front_matter` writes
`model-index`, which the Hub renders as an evaluation widget.

**Seed groups are a naming convention.**
`byosm/report/seeds.py`: experiments named `<recipe>-s<N>` form one recipe group, and a group of at least `MIN_SEEDS`
(2) seeds scored on the same suite version gets a mean and sample sd in the `report:<name>:seeds` block. A recipe scored
on several versions gets one row per version (`@<fingerprint>`); a seed alone on its version stays out. Rows marked with
a dagger are listed last and never ranked against the promoted version. Seed noise is how large a gap must be before
one recipe beats another; a one-seed difference smaller than that sd means nothing.

**One licence default, private by default.**
`byosm/config.py` `PUBLISH_LICENSE`: everything published is CC BY-NC 4.0 (`cc-by-nc-4.0`), the default for
`byosm publish`, `byosm auto` and `byosm.auto.run`. `byosm publish` creates a private repo unless `--public` is passed;
autopilot publishes only a winner, always private, to be flipped public on the Hub after review. Visibility is only set
when a repo is created, so a private publish into a repo that already exists as public stops with an error instead of
uploading publicly (`byosm/hub/upload.py`).

## Deployment

**Jobs and models run in separate containers.**
`docker/`: the app runs any allowed `byosm` command as a subprocess with the workspace keys in its environment, so it
gets its own container that sees only `/workspace` and `/cache`. A served model needs none of that: it gets its own
image with no torch, no keys and no workspace, so a bug or an exploit in one model cannot reach the pipeline, the keys
or another model, and each model scales and restarts on its own.

| image | built from | contains | runs as |
|---|---|---|---|
| `byosm-app` | `docker/app.Dockerfile` (repo root as context) | python 3.11 slim, CPU torch, byosm with the `app` extra, the UI built by node 20 in a first stage | uid 1000, `byosm app --host 0.0.0.0 --no-open` on 8765 |
| `byosm-<name>` | `docker/model.Dockerfile` (the export folder as context), via `byosm containerize <name>` | python 3.11 slim, the bundle's own `requirements.txt` (numpy, onnxruntime, tokenizers for text), the bundle | uid 1000, `byosm_rt.py serve /model --host 0.0.0.0` on 8800 |

**The UI ships as package data.**
`pyproject.toml` `[tool.setuptools.package-data]`: `byosm/app/static` is built, not committed, and a plain (non-editable)
install drops non-Python files unless they are listed, so the app image would serve no UI without it.

**A container needs a stable token and its own Host names.**
`byosm/cli/local_app.py`: by default `byosm app` makes a new random token each start and accepts only `127.0.0.1` and
`localhost` as Host header, against DNS rebinding. `--token` (or `BYOSM_APP_TOKEN`) fixes the token, which is then never
printed; `--allow-host` (repeatable, or `BYOSM_APP_HOSTS=a,b`) adds the names a reverse proxy or another container uses.
The bind address is always allowed too.

**Compose is hardened and local by default.**
`docker/compose.yaml`: both services are read-only with `/tmp` as tmpfs, drop every capability and set
`no-new-privileges`; ports bind to `127.0.0.1`, so exposing them is a reverse proxy's job. The app mounts the workspace
(`BYOSM_WORKSPACE`, default the repo) and a `byosm-cache` volume for downloads, embeddings and OCR, and reads the
workspace `.env` as `env_file`. `BYOSM_APP_TOKEN` is required. The model service is in the `models` profile and uses the
image `byosm containerize intent` built, so `docker compose up` starts only the app. Gotcha: inside the container
`localhost` is the container; an Ollama on the host is `http://host.docker.internal:11434/v1` in `byosm.yaml`.

## README results

**README tables are generated, never typed.**
`byosm/report/`, `byosm/cli/report.py`: `byosm report` rewrites every block between `<!-- report:<name> -->` and
`<!-- /report:<name> -->` from `specialists/*/board.json` (summary, one table per specialist, and seed groups), and
appends a missing block at the end. The notes around the tables are written by hand and carry no numbers. Same idea as
the model card: README numbers cannot drift from the results files.

**Freshness is checked, not trusted.**
`byosm report --check` writes nothing and exits 1 when a block is out of date or missing, or when a block is unknown or
repeated (`tests/test_report.py`). Run it before committing results; run `byosm report` to fix it.

**The dagger marks a different suite version, not an older one.**
`byosm/report/cells.py` `stale_ids`: a row is marked when its fingerprint on that suite differs from the promoted
model's, whichever is newer. On intent the promoted `final-s1` is the only `b77` row on its version, so every other row
carries the dagger. `byosm eval` rescores them onto one version.

## Systems

**A system is a team of ordinary specialists.**
`byosm/system/spec.py`, `nodes.py`: `systems/<name>/system.yaml` lists nodes (a router that picks the action, a tagger
that finds its arguments), the tools they map to and the system suites. `byosm system build` turns each node into a
normal specialist under `specialists/`, with its own board, so every node is proved alone before the team is. The node
spec comes from its dataset, not from an LLM: labels are the dataset's class names, or `O` plus `B-`/`I-` tags for every
argument type found in its markup (`source_labels`); the role sets the task type, metric and default backbone; the
source's train and validation splits train it and its test split is the frozen suite.

**The system suite is materialised once, from the benchmark.**
`byosm/system/frames.py` `materialize`: node suites each read and shuffle their own rows and drop unlabeled ones, so
their calib/test halves need not line up. The system reads the benchmark's split itself, keeps the raw request as the
input and the gold action plus `(type, words)` arguments as the answer, and splits calib/test with the same every-fifth
rule as specialist suites. Every rival is scored on exactly these rows.

**Exact frame is the headline score.**
`units`: a request counts only if the action and the whole multiset of arguments are right (types exact, words compared
after `normalize`), which is MASSIVE's exact-match accuracy. Action accuracy and argument F1 are reported beside it.
Each request's hit is its bootstrap unit; `byosm/resample.py` is the same paired bootstrap the specialists use.

**The LLM rival does the whole task alone.**
`byosm/system/rival.py`: it gets the action list and the argument types and returns frames for ten requests per call.
A missing item is an empty frame and counts as wrong, the same as a wrong answer.

**One runtime runs the team.**
`byosm/rt.py` `System`: the bundle is `system.json`, one exported specialist per node under `nodes/` and a single
`byosm_rt.py`; `predict` on a system folder returns the action, its arguments and the tool call. `act` imports the
tool's `call` (`module:function`) and runs it. Tools are filled by argument type; a missing argument is `None`.

**The text gate fires once, when it is sure for long enough.**
`byosm/rt.py` `TextGate`, `SystemStream`: each keystroke runs the router on the text so far; the gate fires when the top
action reaches `min_confidence` for `stable_for` keystrokes in a row, and never changes its mind. `finish` runs the
tagger on the whole text and keeps the fired action. Gotcha: one gate, one runtime class; the eval replays the same
`TextGate` over cached router traces, so offline and live behaviour cannot drift.

**The gate is tuned on calib, against waiting for the end.**
`byosm/system/streaming.py` `tune`: a grid over `CONFIDENCE_GRID` x `STABLE_GRID` picks the gate that fires earliest
while wrong actions per 100 stay within `stream.max_extra_wrong` of the end-of-request router. `system eval` writes it
to `systems/<name>/gate.json` and the bundle's `system.json`. Traces (the router's top action and confidence after
every keystroke) are cached under `evals/<suite>/trace-<fingerprint>.json`. When no gate meets the allowance, the
action waits for the end.

**Prefix training is a train setting, not new data.**
`TrainCfg.prefix_rate`, `byosm/text.py` `prefix`: each epoch that share of text inputs is cut at a random keystroke and
keeps its label, so the data key and the data folder stay the same. Taggers keep the tags of the words that remain.
It is off by default and draws from the RNG only when on, so seeded runs without it are unchanged.

**Inline markup is a source format, not a new field.**
`HFSource.format`: `markup` means the label column holds `[type : words]` spans. It reuses the existing field
because a value is clearer than a flag next to it.

**A bare sentence runs the autopilot.**
`byosm/cli/base.py` `PromptFirst`: an argument that is not a command is treated as `byosm auto <sentence>`. The compiler
names the task (`free_name` appends `-2`, `-3` when the name is taken) and infers the modality and task type.

## CLI and configuration

**LLM roles fall back to `default`.**
`byosm/config.py`, `byosm/llm/__init__.py` `_role_conf`: roles are `spec`, `generate`, `judge`, `eval`, `analyze` and
`baseline`; any role can use any provider. `anthropic` uses the official SDK; `openai` covers any OpenAI-compatible
server (Ollama, vLLM, LM Studio, OpenRouter). Bulk generation is the expensive part, so a cheaper `generate` model is
usually fine. A role on the same endpoint as `default` (same `provider` and `base_url`) is merged over it key by key,
so `eval: {model: other}` keeps the provider, URL, key and `reasoning_effort`; a role that sets `api_key` or
`api_key_env` drops the default's credentials first. Gotcha: only `openai` accepts `api_key`; `anthropic` silently
ignores it and reads `api_key_env` or `ANTHROPIC_API_KEY`. A role on another endpoint inherits nothing, so it must name its
own provider, model and credentials. `register_provider` plugs in other backends.

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

**Codex runs as a sealed, read-only `codex exec`.**
`byosm/llm/codex_provider.py`: the `codex` provider pipes the prompt on stdin (`codex exec -`) from an empty temp dir
(`-C`), with `--output-schema` (the `_strict` schema, so the final message is the JSON) and `-o` (the final message is
read from a file, not scraped from progress output). Isolation flags, taken from the openai/codex `exec` CLI source and
the non-interactive docs: `--sandbox read-only` (no writes, no network from shell commands), `-c approval_policy="never"`
(nothing waits for a human), `-c web_search="disabled"`, `--ignore-user-config` and `--ignore-rules` (no user MCP
servers, profiles or exec rules; auth still comes from `CODEX_HOME`), `--ephemeral` (no session files) and
`--skip-git-repo-check` (the temp dir is not a repo). The system prompt goes in as `-c developer_instructions=...`, a
JSON string, which is also a valid TOML string. With no `model` the CLI picks its own default. Text only, like
`claude-code`: images raise `LLMError`. Not yet run against a real Codex install; the tests mock `subprocess.run`.

**Provider detection lists every known provider, found or not.**
`byosm/llm/detect.py` `detect`, `GET /api/llm/providers`: `KNOWN` names each provider once (label, kind, install hint,
probe), and the probes run in parallel with a 3 s timeout each. Agents are found on `PATH` (`claude`, `codex`), local
models by their listing endpoint (Ollama `/api/tags`, LM Studio `/v1/models`), hosted APIs by a key from `secret`
(environment or `.env`), whose `/v1/models` then fills the model list. Each entry carries a role `config` ready to paste
into `byosm.yaml`; LM Studio and Ollama use the `openai` provider with a `base_url`, so they need no class of their own.
A hosted config names `api_key_env`, never the key, so no key value leaves the server. A provider that is not found has
`available: false`, `config: null` and a one-line `hint`, so the app can show it greyed out. Gotcha: `anthropic` and
`openai` read `api_key_env` from the process environment; app jobs get `.env` merged in (`byosm/app/jobs.py`), but a
plain CLI run with a key only in `.env` is detected yet must export it.

**Retries are bounded.**
`byosm/llm/base.py` `_retry`: JSON parse errors and `LLMError` retry at once; other errors retry only when they look
transient (rate limit, overload, timeout, connect), with 2, 4, 8 s backoff. `pmap` reports failed items through
`on_error` and skips them.

**Commands register by import.**
`byosm/cli/__init__.py`: importing the command modules (`define`, `iterate`, `report`, `ship`, `system`, `train`) is what
registers their `@app.command`s, so a new command module must be imported there.

**Commands that just scored print the detailed board.**
`byosm/cli/train.py`, `byosm/cli/iterate.py`: `build`, `eval` and `auto` call `board(name, detail=True)`, so the CI and
P/R columns appear after a run; plain `byosm board` hides them unless `--detail` is passed.

**App jobs report progress as typed events on stdout.**
`byosm/events.py`, `byosm/app/jobs.py`: a job is a `byosm` subprocess, so library code reports progress by printing
one `@event <json>` line, and only when the app sets `BYOSM_EVENTS=1`. Plain CLI output stays unchanged, the library
needs no import of the app, and the runner publishes each line as a `metric` event and stores it next to the job's log
in `.byosm/app.sqlite`, so history and charts survive a restart.

**Feature routes are routers.**
`byosm/app/panel_routes.py`: a feature with its own endpoints is an `APIRouter` that `create_app` includes with the
token dependency, so `server.py` stays small. The panel endpoint reshapes each review row so teacher votes sit under
`votes` and cannot collide with `input`, `panel` or `sure`; a review is validated against the spec's labels before it is
written.
