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

**One database holds every specialist.**
`byosm/project.py` `Project` reads and writes the database in `byosm/db/` through SQLAlchemy: SQLite in
`.byosm/byosm.sqlite` by default, Postgres when `DATABASE_URL` is set, one schema for both. Tables: `specialists` (spec,
promoted experiment), `experiments` (snapshot spec, `meta`, the proposal it applied, dev errors), `results` (one row per
predictor and suite, the scoring cache and the board), `reports` (overfitting check, proposals, autopilot decisions,
dataset suggestions, test-set results, keyed by kind) and `rows`. Only experiments with a `meta` count as trained. The
board shows the results of trained experiments and the spec's baselines on the spec's suites; results of a removed
baseline stay as a cache. Files hold only what is a file by nature: weights in `specialists/<name>/experiments/<exp>/`,
the ONNX bundle in `export/`, the audio clips and page images that rows point to, and download caches.

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
`byosm/data/judge.py` `judge`: the `judge` role relabels each generated row blind, and only rows where both labels
agree are kept. Label noise caps the precision a tiny model can reach, and this is cheap insurance. It is a second
opinion only when `judge` is configured as a different model. `judge_labels` returns the judge's own label for each
input in order, so a step that needs the judge's label rather than its agreement, such as relabelling mined
near-miss pairs, asks it the same way; a row the judge drops comes back as `__drop__`, a failed batch as none.

**Refinement learns from dev errors.**
`byosm/pipeline.py` `refine`, `byosm/data/synth_text.py` `refine_jobs`: the `analyze` role reads the dev errors and
returns failure patterns with data recipes; the `generate` role then writes targeted training rows from them (filtered
by `judge` when it is on), deduped against train, dev and test. The refine prompts come from the patterns and do not
read `hard_cases`. The refine loop itself never sees test rows, but dev is the builtin suite's calib split (see "The
builtin calib split is also the training dev split"). Each round retrains from scratch on the grown `train.jsonl`
and overwrites `model.pt` (`byosm/train/outputs.py` `save_run`), so `build` copies the best round's weights aside and
keeps the round with the best dev score, as epoch selection does within a round (`kept_round` in the experiment meta).
A tool-call gate run showed why: round 0 reached dev 0.871, round 1 fell to 0.833 and would have replaced it.

**Refinement is opt-in and stops at the first round that does not help.**
`train.refine_rounds` defaults to 0, so a build writes no LLM rows unless its spec or a proposal (`refine-more`,
`hard-pairs`) asks for rounds. Refinement used to run in every text classifier build below its target, whatever the
build was for, and it rarely helped: on the tool-call gate dev went 0.633 to 0.467, 0.600 to 0.557 to 0.442 and 0.871 to
0.833 to 0.798 across rounds, and a Banking77 ablation found it neutral. When rounds are on, `build` stops after a round
that does not beat the kept one instead of writing more rows from a worse model's errors.

The rules that ask for more data read their own evidence (`byosm/improve/registry.py` `plateaued`, `refine_hurt`).
`more-data` is proposed only while dev is still rising or noisy, not after it plateaued: its text used to say "dev
plateaued at 0.86, so more of the same data may help less" and propose doubling the data anyway. `refine-more` and
`hard-pairs` are dropped once the last build's refine rounds left dev no higher than round 0 (`refine_gain` in the
diagnosis). The analyst also flags dev rows whose gold label looks wrong (`suspect_gold_ids`); when at least half the
errors are flagged (`MISLABELLED_SHARE`), refine writes nothing and says to check the label definitions, since new
training rows cannot fix a disagreement about the labels. Before, the flags were requested and then ignored.

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

**LLM-written data is sized to the time it is allowed.**
`byosm/data/budget.py`: the first `MEASURE_CALLS` generation calls are timed, the remaining calls for train, dev and test
are estimated at that pace, and when they would overrun the allowance every split is cut by the same factor, down to
floors (`MIN_TRAIN_PER_LABEL` per label, `MIN_DEV_ROWS`, `MIN_TEST_ROWS`) that win over the budget. The allowance is
`data.max_minutes` for `byosm build` and `DATA_BUDGET_SHARE` of the time left for the autopilot (`byosm/auto/state.py`
`Session.data_seconds`, the only place the autopilot's deadline is turned into a data budget). The pace is wall time per
call times the calls that can really run at once (`Observed.parallel`: one for a local server), and includes the
judge's pass over the timed rows. `fit.json` in the data folder keeps the planned, written and
actual sizes, the pace and the budget, and the model card says when the set was shrunk. Gotchas: dev and test are
assumed to run at the train model's pace even when the `eval` role is another model; `max_minutes` is left out of the
data key, so a shrunk set is reused by later builds with the same plan (delete the data folder to regenerate);
refinement rounds are not budgeted.

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
borderline tips), and the request itself as a `data-request` report in the database. Test rows are sized so accuracy is measured to within 3 points at
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
`byosm/train/rows.py` `split_rows`: a random 10% of train (at least one row) is held out with the run's seeded RNG, so
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

**A local LLM server gets as many calls as free memory allows; a hosted one is not limited.**
`byosm/llm/__init__.py` `_gate`, `byosm/resources.py`: every role config shares a semaphore per endpoint and model;
endpoints on `LOCAL_HOSTS` allow one call per `LOCAL_CALL_GB` of free memory (at least one), hosted ones are
unlimited (a job still sends at most `REMOTE_PARALLEL` at once), and `max_parallel` on a role overrides both. Jobs follow
the same idea: the app starts a queued job when nothing runs or at least `JOB_GB` is free, up to one per CPU. Gotcha: Ollama serves requests one after another, so eight parallel clients made every request wait in its
queue until the ten-minute timeout, fail and retry, and two jobs sharing it made no progress for hours. The gate only
works inside one process; two jobs on the same local server still share it.

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

**The overfitting check reads the board and scores only a train sample.**
`byosm/evals/overfit.py`, `overfit_checks.py`: five findings, each `ok`, `warn` or `fail`, and the verdict is the worst.
The train score is the experiment scored by the same `score` as the board, on 1000 of its own training rows (the rows
`split_rows` gives with the spec's seed) with the primary suite's calib as the threshold split, so train and test use
the same decision rule. Calib vs test compares the calib bootstrap CI (from the stored calib units) with the test CI.
Overlap indexes the training rows with `ShingleIndex` (the same exact and 5-word-shingle test as `dedupe`) and rescores
each suite's test units without the overlapping rows: a leak whose removal barely moves the score is only a warning.
Selection pressure is the expected lift of the best of N equal experiments picked on calib (Blom's approximation of the
expected normal maximum times the calib standard error); it warns when that lift exceeds the winner's lead over the
median of the others. The learning curve reuses `training_summary`. The report goes to `specialists/<name>/overfit.json`,
keyed by a hash of the board and the experiment's `meta.json`; the train score has its own key (meta, primary suite
fingerprint, sample size), so a new board result re-reads the evidence without scoring the train sample again.
It is opt-in after a build (`byosm build --overfit`), because scoring the train sample took 4 to 27 s for the text
specialists, 17 s for vad and 79 s for doctype on CPU (measured); the autopilot runs it once, before export.
Gotcha: overlap compares text only; for audio and document inputs it is listed under `skipped`. For a builtin primary
suite, calib is the dev split that picked the best epoch, so calib there is optimistic by construction.
The fix follows the finding (`byosm/next_fixes.py` `less_overfit`): a dev curve still rising at the last epoch with no
train/test gap is underfitting, so the step trains for twice the epochs; a gap or a dev drop after the best epoch
halves the learning rate and stops at the best epoch; calib/test drift alone gets no training fix, since a slower
model does not make the two splits alike. A fix whose build never finished is offered again as a build of the same
experiment instead of disappearing: tool-choice's fit-v1 was interrupted and its step was never shown again.

**A written test set keeps the inputs' layout.**
`byosm/evals/testset.py` `with_layout`: when the spec has no `input_description` (a specialist trained from a file),
the writer is shown three real inputs and told to keep their separators and field order. Without them each writer made
up its own layout: tool-choice trains on "request, then `Option: name: description. (params)`", and two written sets used
"`Option: ...` then `Request: ...`" and "`Offered tool: ... — ...` then `Request: ...`". On those rows the 11M model
answered no_fit to nearly every fitting tool at 0.98 confidence (0.44 and 0.52 accuracy), a test of a format it never
sees in use rather than of the task.

**An LLM-written test set is a check, never the benchmark.**
`byosm/evals/testset.py`, `byosm testset NAME`: the `eval` role writes `--per-label` examples per label with the same
adversarial prompts as the builtin test split (`synth_text.generate`, now with a `labels` filter so top-up rounds ask
only for labels still short), deduped against train, dev and test and judged when `data.judge` is on. Rows are kept per
label and split calib/test like `hf` suites (every 5th to calib), frozen under `evals/<suite>/` as kind `generated`,
and only the promoted experiment and the baselines are scored on it. The suite records `written_by` and `judged_by`
(model name from the role config, never keys or URLs), so a result reads "written by qwen3.8:27b". It cannot be
primary (`EvalSuite` validator) and `primary_suite` skips it when no suite is marked, so the results page never reports it.
Gotchas: the writer can favour models of its own family, and the LLM baseline may be the same model that wrote the
items; read its score there as self-agreement. Frame tasks are refused. An existing suite name is refused, so a rerun
needs `--suite`.

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
The time budget counts the session's own work: its deadline moves out by every second spent queued for the GPU
(`train.gpu.waited_seconds`), the per-experiment estimate leaves those waits out, and the experiment budget counts this
session's experiments, not every experiment the specialist ever had. Before, the tool-call gate's autopilot spent its
6 hours building v1, 3.1 of them queued, and stopped without trying a single proposal.

When the loop stops without a win, the autopilot climbs one rung (`byosm/auto/escalate.py`): the smallest larger
backbone in `spec/backbones.py` that it has not tried and that speaks the spec's languages, built as an ordinary
`bigger-backbone` proposal so it never runs twice. With no larger backbone left it saves an `unreachable` report with
the score, the target and every backbone tried, and the "Meets the target" check says "unreachable at this budget".
Before, a stalled run just distilled and exported, and the next run started the same cheap fixes again.
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

## Results page

**Result tables are generated, never typed.**
`byosm/report/`, `byosm/cli/report.py`: `byosm report` rewrites every block in `docs/results.md` (`RESULTS_PAGE`) between `<!-- report:<name> -->` and
`<!-- /report:<name> -->` from `specialists/*/board.json` (summary, one table per specialist, and seed groups), and
appends a missing block at the end. The notes around the tables are written by hand and carry no numbers. Same idea as
the model card: published numbers cannot drift from the results files.

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

**Forecaster training resumes after a restart.**
`byosm/series/checkpoint.py`: the patch transformer saves its weights, optimiser, schedule and random state every 5000
steps to the cache, under a key made from the recipe, the step count and the training series. Retrying an interrupted
job finds the checkpoint and continues from that step with the same result as an uninterrupted run; the checkpoint is
deleted when training ends. A reboot once cost a sales run at step 87,000 of 96,000 its whole night of GPU time.

**App jobs outlive the server.**
`byosm/app/detached.py`, `byosm/app/jobs.py`: each job starts in its own session with its output in
`.byosm/jobs/<id>.log`, and a small wrapper writes the exit code to `<id>.exit` when it ends; the pid is stored with the
job. The runner tails the log instead of a pipe, so stopping or restarting the app never kills a job. On startup it
reattaches to jobs whose process is alive and skips the lines and events already stored, finalises jobs that left an
exit file, marks failed only those whose process is gone without one, and requeues jobs that never started. Cancel
sends SIGTERM to the whole process group. This gives durable jobs without a workflow engine such as Temporal: one
machine, one server per workspace (`byosm/app/lock.py`), and SQLite plus two files per job as the source of truth.

**Feature routes are routers.**
`byosm/app/panel_routes.py`: a feature with its own endpoints is an `APIRouter` that `create_app` includes with the
token dependency, so `server.py` stays small. The panel endpoint reshapes each review row so teacher votes sit under
`votes` and cannot collide with `input`, `panel` or `sure`; a review is validated against the spec's labels before it is
written.

## Live microphone tests through LiveKit

**The listener is a job, not part of the app server.**
`POST /api/specialists/{name}/live` starts `byosm listen <name> --room <room>` as an ordinary app job and returns a
browser token for the same room. The listener joins as `byosm-listener`, takes the first audio track, resamples it to
16 kHz in 32 ms chunks with `rtc.AudioStream`, and sends one data packet per chunk on topic `vad`. Running it as a job
gives it the jobs page, its log, Cancel and a separate process for free. It leaves when the browser leaves, after
20 s if nobody joins, and after 15 minutes at most.

**Both models see the same frames.**
Each chunk goes to the exported bundle's `stream().step` and to Silero's ONNX model (512-sample windows, the model
`livekit-plugins-silero` ships), so the browser plots the two probabilities on one clock. Ours speaks through its
own hysteresis from the manifest; Silero's state is its probability against LiveKit's default 0.5 activation. Timing
is CPU time per 32 ms chunk inside the listener.

**Configuration.** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in the workspace `.env`, falling back to
`livekit-server --dev` (`ws://localhost:7880`, `devkey` / `secret`). `LIVEKIT_PUBLIC_URL` is the address the browser
uses when it differs from the one the listener uses (a container, a reverse proxy). The endpoint answers 503 with
the command to run when no server listens there.

## Health score

Nine checks, each graded pass, warning, fail or not measured, and weighted: lead over the best rival (paired
bootstrap on test, 3), target met (2), no overfitting (2), holds on suites other than the primary (2), and 1 each for
enough data, a precise score, no weak label, latency within budget and a current export. The score is the weighted
share of credit (pass 1, warning 0.5). Production ready means 80 or more and no check failing or unmeasured: an
unmeasured check is not evidence of quality. Calibration is not graded because results store labels, not confidences.
The rival bootstrap takes seconds on large test sets, so it is cached by the two results' fingerprints.

## Adopting an experiment's data as the benchmark

An experiment that changes the data plan builds its own splits, but the board scores every model on the specialist's
builtin suite, the held-out split of the base spec's data. When that split is below `MIN_TEST_ITEMS` or
`MIN_CALIB_ITEMS`, every model can tie (turn-completion: 9 test rows, calib 0.000 for all), so selection never moves.
`byosm adopt-data NAME EXP` (`byosm/held_out.py`) copies the experiment's data plan into the spec, rescores every
experiment and baseline on the new held-out split, and promotes the best on calib. A build does this on its own when the
experiment it just trained is the one with a big enough held-out set, and Next steps offers it otherwise. Older experiments may have trained on rows that land in the new test split;
the overfitting check's overlap finding shows it.

## One limit on parallel LLM calls

Every loop that makes more than one LLM call (writing data, the label judge, scoring with an LLM rival, a team's LLM
rival, the data check's judge, mapping several Hub datasets) runs its calls through `pmap` with
`parallel_calls(cfg)`. That is unlimited up to `REMOTE_PARALLEL` unless a run sets `--parallel`; the real limit is the
per-endpoint gate, which the Trainers page sets ("Parallel calls") and which follows free memory for local models. There
is no second, global `concurrency` cap any more, because it silently throttled remote models to a handful of calls.

## Artifacts on Hugging Face

The database holds every record; weights and ONNX exports are files. `artifact_repo` in `byosm.yaml` names a private
Hugging Face model repo that mirrors them under the same `specialists/<name>/...` paths (`byosm/artifacts.py`).
`byosm store push NAME` uploads a specialist's weights and export, `byosm store pull NAME` downloads them, and loading an
experiment whose `model.pt` is missing pulls just that experiment. Push refuses a public repo. Nothing is uploaded
unless `artifact_repo` is set.

## One rule for "it really is better"

Promotion after a build, the autopilot's keep/revert and the health check's "beats the best rival" all use
`sure_lead` in `byosm/evals/selection.py`: a lead counts only when a paired bootstrap keeps it in `SURE_LEAD` (95%) of
resamples. Promotion used to compare calib point estimates, which let turn-completion's v17 replace v15 on a calib lead
that held in 88% of resamples (interval -0.77 to +0.94), while v15 stayed ahead on test even at v17's best threshold.
With a bound like recall at 98% precision on a 391-row calib split, a handful of rows decides the score.

## Fine-tuned rivals

Jev and an LLM are not the bar to beat when someone already fine-tuned a model on the same benchmark. `byosm rivals NAME`
(`byosm/finetuned_rivals.py`) asks the web-research LLM for the strongest public models fine-tuned on the primary
benchmark or the labelled training source, checks each id exists on the Hub, then adds models tagged on the Hub as
trained on that dataset. It skips models above 1B parameters, keeps up to two whose labels name ours, or whose
`LABEL_n` outputs map onto ours on at least 80% of our dev rows, and scores them with the same code. Next steps offers
it once per specialist with a public benchmark.

## Ablations decide what becomes a default

A training idea (LeJEPA pretraining, active data, a label model over several LLM teachers) becomes a default only after
`byosm ablate NAME --set path=value --seeds 0,1` (`byosm/ablation.py`). For each seed it builds a matched pair from the
same base experiment, identical except for the change, and compares the pair with a paired bootstrap on calib and on
test. The verdict is "better" only if every seed's calib lead holds in 95% of resamples and the test score agrees,
"worse" if every seed is surely behind, and "not sure" otherwise. Each verdict is saved as an `ablation:<change>` report.
Run it on at least two public benchmarks before switching a default.

## Refinement rows belong to their experiment

Refinement used to append its targeted examples to the shared `train` split of a data key, so every later experiment on
that key trained on them silently (turn-completion's v20 trained on v15's, v17's and v18's examples while its spec
said nothing about them). Now `pipeline.refine` writes to `refine-<exp>`, `fit` trains on the splits it is given, and
each experiment records them as `train_splits` in its meta; the overfitting check, the diagnosis and the model card read
exactly those splits. The rows a build's own rounds write go to `refine-<exp>-rounds`, which `build` empties first, so a
retried build no longer trains on the rows of the attempts before it: the gate's v1 had collected 2,878 such rows from
six builds, five of which failed. Rows that Keep training writes before the build stay in `refine-<exp>`. The mixed turn-completion rows were split back by position, which the recorded round sizes and
row sources confirm.

## Model cards: facts by code, prose by an LLM, numbers checked

`byosm/hub/card_writer.py` builds a facts object (task, labels, results with intervals, rivals with links, the splits
the model trained on, refinement rounds, export size and dependencies, limitations) and asks the `card` role for prose
in the structure of livekit/turn-detector's card. Results tables, test-set tables, usage and reproduce blocks stay
exact: the LLM places them through `{{RESULTS}}`-style placeholders. Every number in the prose must match a number in
the facts or the exact blocks (rounding, percent and millions allowed); otherwise the LLM gets one retry with the
offending numbers, and after that the template card is used.

## Archive instead of delete

`byosm/archive.py`: archiving keeps everything a specialist built and only hides it from the list (the "Archived"
filter on the specialists page shows it again, and Restore brings it back). The archived names are one workspace
record, so models, teams, forecasters and event risk can all be archived without a schema change. Deleting a
specialist also takes it off the archive list.

## Publishing with a preview

`byosm/hub/preview.py`, the publish dialog: before pushing, the dialog stages the bundle as a dry run and shows the
card exactly as it will be published, the files and their size. The card can be edited by hand or by asking the `card`
role, whose edit is refused when it adds a number that is not already in the card (the same `unsupported` check that
guards the generated prose). The edited card is saved next to the staged files and passed to `publish --card`. "Test
the bundle" loads the exported files with their own runtime, scores up to 200 test inputs and answers an input typed in
the dialog, so what is tested is what is pushed. "Include the dataset" maps to `publish --data/--no-data`.

## Publishing the data next to the model

For text specialists, `byosm publish` also uploads `<repo>-data` as a Hugging Face dataset (`byosm/hub/data_release.py`):
the splits the published experiment trained on (with the split each row came from), the calib and test splits, and
every test set, with each model's answer on every test row. Each row keeps its `source`; the dataset card lists the
licence of every public source from its own card. The model card's metadata gets `language`, `base_model` and points
its evaluation result at that dataset, so a reader can rescore any rival on the same rows.

## Proposals must be buildable

The analyst sees the rivals on the board and may propose one of them as the new backbone. Some Hub models ship only
ONNX weights, which run as rivals but cannot be fine-tuned, so a job that tries crashes after its data step. Before a
proposal is applied, `improve.paths.untrainable` lists its Hub backbones that have no PyTorch weights, and the proposal
is dropped with a log line. The check is one file listing per repo; when the Hub cannot be reached it lets the proposal
through, since loading from the local cache may still work.

## A proposal runs once

`byosm/improve/proposals.py` `apply`: a proposal whose edits already ran is refused (`AlreadyTriedError`, or
`improve --force` to run it again), and one whose build never finished reuses that experiment instead of creating a
new one. Every attempt counts, built or not (`Project.attempts`): `next_steps.tried_proposals`, the autopilot and the
analyst's history all read it, and the history says when an attempt never finished. Before, only experiments with a
finished training counted, so a failed or interrupted run looked untried: the tool-call gate's drop-xlam proposal ran
three times and v4, v5 and v6 have identical specs.

Setting `data.hf_sources` to an empty list on a spec that had sources also clears `data.hf_queries`. Without that, data
resolution treats an empty source list as "search again" and picked the same sources back: the drop-xlam runs still
trained on xlam.

## A crashed job keeps its error

The job wrapper writes the exit code in a `finally` block. Python prints an uncaught traceback only after that block, so
the runner saw the exit code, read the log one last time and deleted it before the error landed. The wrapper now prints
the traceback itself before writing the exit code.

## Labels derived from content

Many useful datasets have no label column: a function-calling corpus says whether the assistant called a tool only in
the assistant's reply. When the mapper finds no label column, the LLM may give `label_rules` instead: `[column, text it
contains, our label]` checked in order, with an optional `label_default`. Columns are paths into nested values
(`conversations.1.value`), so the input can be the user turn while the rules read the reply. The prompt forbids an input
that contains the text the rules look for, since the model would learn the answer from the input.

## When the LLM hits its limit

A subscription or API limit should pause a job, not kill it. Every LLM call goes through `llm.recover.through_limits`.
A limit error (recognised by its message, never retried by the provider's fast retry loop) follows the `on_limit` mode
in `byosm.yaml`, which the app can change while jobs run:

- `wait` (default): sleep until the reset time the message gives, or with backoff when it gives none, then retry. A
  `limit` event carries the time, so the app shows a banner with **Retry now** and **Switch to local**.
- `local`: send this and the following calls to the local model (`on_limit.local`, Qwen through Ollama by default).
- `fail`: raise as before.

The waiting call re-reads the mode every minute, so switching to local or pressing Retry now reaches jobs already
waiting. A failed job also offers Retry and Retry with local LLM. A job's chosen LLM now replaces every role, not only
the default one, so "run this with X" means every part of the job uses X.

## What a specialist cost, and when it pays off

The Costs page prices what BYOSM measured, not estimates. Each LLM call event carries its job and token counts; a job
names its specialist (from its arguments, or the `target` event an autopilot emits). Calls are priced from LiteLLM's
public price table, cached for a day under `~/.cache/byosm`, and split into building (data, labels, analysis) and
evaluating (rival answers, test sets). Training power is measured training time times `costs.watts` at
`costs.usd_per_kwh`, both settings in `byosm.yaml`. The LLM's cost per item is its measured tokens on the rival runs
divided by the items those runs scored; ours is the measured CPU time per input at the same power price. Break-even is
the build cost over the saving per item. Calls made through Claude Code include its own prompt, which inflates build
and per-item cost alike, so the break-even is steadier than either dollar figure.

## A cap on trainings sharing the GPU

Seven jobs training on the Apple GPU at once froze a 64 GB M5 Max until the kernel watchdog rebooted it ("no checkins
from watchdogd in 91 seconds"). Jobs still run in parallel, but `train.gpu.gpu_slot` lets only a few of them pretrain
and train on `mps` or `cuda` at a time. The machine decides how many: one per 8 CPU threads and per 16 GB of GPU-usable
memory, at most two (two on an 18-thread, 64 GB M5 Max); `gpu_jobs` in `byosm.yaml` overrides it. The slots are
`flock`ed files under `.byosm/gpu`, so the operating system releases a slot when its job dies, and a waiting job logs
that it waits. LLM work (writing and checking data, analysis) needs no slot and keeps running alongside.
GPU memory on Apple silicon is wired memory, invisible in process sizes: one training was holding 36.7 GB while its
process showed 0.5 GB. Each training therefore caps PyTorch's MPS allocator at 90% of the GPU's recommended working set
divided by the slots, so two trainings cannot take more than the GPU has, and one that needs more fails with an
out-of-memory error instead of freezing the machine. The machine card shows the wired memory. A training also waits for its slot until
16 GB of RAM is free, so a third heavy job cannot push the machine over, and the app shows a banner while less than 8
GB is free. When a training still runs out of its share (ModernBERT-base with LeJEPA's two views at batch 32
needed 24 GB), the batch is halved. On CUDA it retries in the same process. On Apple GPUs it does not: after an MPS
out-of-memory error the next attempt in the same process failed inside Metal ("command buffer exited with error
status") and hung, holding its slot for eleven hours while every other job waited. So the job stops with a clear error,
the halved batch is remembered per modality and backbone in `.byosm/gpu/batches.json`, and the next run, a fresh
process, starts there. The run's metadata records the batch it trained with.

## A stuck job cannot hold the queue

The runner stops a job that writes nothing for `stall_minutes` (90 by default in `byosm.yaml`) and marks it failed, so
Retry is one click away. Waiting is not silence: a job waiting for the GPU or for an LLM limit to reset sends a
heartbeat event while it waits.

## Each model in its own container

`byosm deploy NAME` runs an exported specialist the way it would run in production, isolated from the app and from other
models: the image from `docker/model.Dockerfile` (python-slim, the bundle's requirements, a non-root user), a read-only
filesystem, at most 1 CPU and 1 GB, published only on 127.0.0.1 at a free port, and labelled `byosm.specialist` so the
app can find it again after a restart. The Deployments page lists them from `docker ps` and `docker stats`, refreshes
every five seconds, and sends test inputs to the container itself, so the answer and its time are the container's own.
`byosm undeploy NAME` stops it; a stopped container leaves nothing behind (`--rm`). A team deploys the same way: its bundle holds
each member under `nodes/`, the image installs the union of their requirements, and the runtime answers
`/v1/<team>/predict` with the team's action and tool call. Every build copies the current `rt.py` into the bundle, so an
image never runs an older runtime than the app that built it.

## Interrupted is not failed

A job whose process vanishes without an exit code, or that receives a SIGTERM the user did not send (a shutdown, a
reboot, the app stopping), ends as `interrupted`, not `failed`: it did nothing wrong. The stall watchdog's kill stays
`failed`. The Jobs page lists interrupted jobs that have not been retried since and retries them all at once, an
autopilot resuming its specialist by name.

## Open data as a source

Useful training data often lives on government portals, not on Hugging Face. `byosm.opendata` searches them through
their APIs, not by scraping: one CKAN client covers data.gov.ie and data.gov.uk (and any other CKAN portal by adding its
URL), and the EU data portal, which aggregates most national portals, has its own search API. Every result comes back
in one shape (title, publisher, licence, date, tabular files only). A file is downloaded once into
`~/.cache/byosm/opendata/<hash of its URL>`, zip archives unpacked, and previewed with a profile of each column from the
first 5,000 rows, so a 47 MB register previews in a second. Downloads run as jobs (`byosm opendata fetch URL`) so they
show progress. The datasets a user keeps live in the database. The Datasets page shows them next to every train, dev
and test set built for a specialist. Two kept tables can be joined: keys are suggested by
matching column names (and key-like names such as `*_num`, `*_id`), then measured as the share of left keys found
anywhere in the full right column, so a suggestion comes with the evidence for it. The joined preview reports the match
rate on the left sample. CSVs are read with the separator sniffed once and the fast parser, which reads the 827,790-row
CRO register's key column in under a second. Training on a joined table is the next step; the web-research agent is for
sources no portal lists.

## Snapshot datasets from registers

A register (companies, licences, permits) is one row per entity with a start and an optional end date. `byosm/tabular/snapshots.py` turns it into a prediction task: for each cutoff year, keep the entities alive on 1 January, label whether they end within the horizon, and add their age. Train, calib and test are different cutoffs, and the last train cutoff plus the horizon must not pass the test cutoff, so no label is seen before the day the model would be used.

A register is a snapshot of today, so any field rewritten after the cutoff can carry the answer. `byosm/tabular/leakage.py` scores each feature alone (one cutoff only, because the same entity repeats across cutoffs and would sit on both sides of a fold) and flags any with AUC of 0.85 or more, or whose missingness differs by 30 points between classes. On the Irish company register it caught two leaks: masking a field changed after the cutoff (only survivors get changed), and the company type, which the Companies Act 2014 rewrote for every company still alive in 2015 and 2016. Collapsing types to their pre-Act family removes it.

## Finding open data by asking

A request on the Datasets page runs `byosm opendata find`. The research LLM (the one with web search) returns three lists: keyword searches for the portals, tables it found on the web with direct file links, and portals not yet known. Every file link is fetched with a one-byte range request and dropped if it does not answer, because LLMs invent plausible URLs. Portals it names are probed as CKAN, then Socrata, at the given URL and at the site root, and saved as sources; anything else is saved as a page the LLM is told about next time. Portal searches match loosely (the EU portal returned aftercare statistics for a food inspections request), so a portal result is kept only if its title or notes share a word stem with the request or the query. When the LLM gives no searches, the request's own content words are used. The request and its results are saved after each step so the page fills in while the job runs.

## Forecast specs

A forecast is saved in the database like a specialist, from a YAML spec: the source file URL (downloaded once into the open-data cache), the start and end columns, the horizon, the cutoff years per split, the features, and derived columns built from a small fixed set of steps (`part`, `first`, `word_count`, `contains`, `keywords`, `frequent`) so a spec never needs code. Saving refuses a spec whose last train labels reach past the first test cutoff. `excluded` names the columns left out and why, and the page shows them next to each feature's AUC alone, so the reasons for leaving data out are visible next to the evidence.

## History features

A spec can name event tables (filings, inspections, payments) by key and date. For each snapshot, every entity gets the number of its events in the year before the cutoff and the days since its last event, using only rows dated before the cutoff. Check how far back an event table really goes before choosing cutoffs: the Irish filing files start in late 2022, so a 2023 cutoff saw a nearly empty year of filings while its test year saw a full one, and the features looked useless. Moving both cutoffs into fully covered years made days since the last filing the strongest single feature.

## Forecasts made from the page

The New forecast dialog reads the kept table's preview: the start date is the date column most often filled, the end date the one most often empty (only ended entities have it), and identifier-like columns (nearly all values distinct) start unticked. Train cutoffs are placed so their labels end on the test year. Text columns that hold numbers become numbers; every other column is a category capped at its 250 most common values, which gradient boosting needs. Features the leakage check flags are left out before scoring and shown in red, so a careless pick cannot produce a headline number. The threshold only catches blatant leaks: on the company register a dialog-made forecast flagged the status columns and `nard` but not the post-Act company type, which still scores 0.65 alone, so review the per-feature table before trusting a result.

## Backtest over test years

One test year can flatter or punish a model by chance of timing. A forecast can list `backtest` years; for each, it is retrained on three cutoffs two years apart whose labels end on that year, and scored on the companies alive on 1 January of it, with the leaks found on the main run left out. Snapshots are built once per year and shared. A spec can also give only `test_year` and the training windows are placed the same way, so the page and the CLI agree. Test years whose outcome cannot be known yet (year plus horizon after this year) are refused. On the company register, the register-only model loses accuracy steadily in later years, which a single 2016 test hid.

## Leaks that predict weakly

A field rewritten for survivors can score a modest AUC alone and still be a leak, because it splits one class cleanly while leaving the rest mixed. The check therefore also measures purity: the share of positives that sit in values of 200 rows or more that are 95% positive, and the share of negatives in values that end at under 2% of the base rate. Above 20% either way the feature is left out. On the company register this catches the raw company type and its code (the old type names are almost all closures, because companies keeping them were the ones about to close) while the pre-Act type families pass.

## Scoring who is alive today

Every forecast run ends by scoring the entities alive on 1 January of this year. The model is retrained on the spec's own training years moved forward by the gap between this year and the test year, so the pattern that was tested (including where history tables are complete) is the one used, and every label it trains on is already known. The page shows how many were scored, how many are expected to end (the sum of their chances), how they spread across chance bands, and the most likely to end with their features in a tooltip. Next to the expected count it shows the predicted and actual rate on the test year, because a sum of chances is only as good as the model's calibration.

## Reasons for a chance

Each of the most likely to end gets up to three reasons: every feature in turn is set to its typical training value (the median, or the most common category) and the fall in the chance is measured; the largest falls of at least one point are the reasons. It needs only the fitted model, costs one prediction per feature, and reads as "this company would be less at risk if this were ordinary". Features interact, so the falls do not add up to the chance; they rank what matters for that row.

## Retrying a training after running out of GPU memory

On Apple GPUs a training that ran out of memory cannot safely try again in the same process: Metal can be left in a state where the next command buffer fails or hangs. Each training attempt on an Apple GPU therefore runs in a fresh child process (spawned, with the training step pickled), while the job keeps the GPU slot. A failed attempt records the halved batch or the next precision and the next attempt starts clean in a new child. An earlier version restarted the whole job with `os.execv`, which replayed every stage of an `auto` job, LLM calls included; the child process limits the retry to the training step.

## Open data in a specialist's training set

When a specialist's data plan searches for datasets, the open data portals are searched with the same terms alongside the Hub. Portal tables become hits with an `opendata:<file url>` id; their size is checked first (a quick request for the length) and files over 100 MB are never offered, because portals often label a 300 MB geodata export as CSV. The picker sees up to eight of them in the twenty results it reads, and is told an open table fits when it has a free-text column and a column that gives one of our labels. A picked table is downloaded once into the open data cache, then `open_split` serves it as a streamed dataset, so the column mapping, label maps and label rules work exactly as for Hub datasets. On a planning-application task it picked two council planning files, read the proposal text as input and derived the labels from the application type.

## Recovering from a diverged training

A model whose outputs turn into NaN raises `DivergedError` at the next dev evaluation instead of failing deep inside a metric. Recovery climbs one step of a precision ladder (auto, which is float16 on Apple GPUs, then bfloat16, then float32), remembers the step for that modality and backbone next to the remembered batch sizes, and restarts the job like an out-of-memory failure. The default stays float16: bfloat16 has float32's range, so it cannot overflow where float16 does, but its shorter mantissa cost a small test model its win over the baseline, and on this Mac a 4096 matrix product takes 2.4 ms in float16, 2.8 ms in bfloat16 and 31 ms in float32.

## Forecast questions on the Home prompt

`byosm auto "<prompt>"` for a new task first asks the LLM whether the request is a forecast (will entities in a register end within some years) or a specialist. A forecast gets a horizon and a few portal searches; the first four readable tables under 100 MB are downloaded and profiled, and the LLM picks the table, its start and end columns, the features known at the start, and the columns that identify an entity. Every name it gives is checked against the table, the test year is the latest whose outcome is known, five backtest years follow, and the forecast is saved and run like one made by hand. The leakage check still runs, so a careless pick is caught there rather than trusted.

## Predicting one entity

Each forecast run keeps the model it scored today with (`.byosm/forecasts/<name>.joblib`), its categories, the features it left out and the typical values used for reasons. A single entity, looked up by its key in the source table or typed in the table's own columns, goes through the same derive, snapshot and history steps as training, so its chance matches the bulk run exactly. Steps that depend on the whole table (`frequent`, which keeps values seen at least 200 times) are skipped for one row; values the model never saw fall into the "other" category or count as missing, as they did in training. An entity that already ended before 1 January is reported as ended rather than scored.

## Publishing datasets and forecasts

`byosm publish-data <name> --repo <model repo>` pushes a specialist's train, dev and test sets on their own, to the same `-data` twin repo the model publish uses, and the Datasets page offers it per specialist once it is exported. `byosm forecast publish <name> --repo <id>` stages the scoring model, the spec, `results.json` and a card generated from the stored results (test AUC with CIs, accuracy by test year, the columns left out and why, the source and its licence) and uploads that folder. The list of scored entities is never staged: it names real companies with a chance of closing, which is not something to publish. Both default to a private repo.

## Deploying to AWS

`byosm cloud <name>` sizes a container from the exported bundle and writes a CloudFormation template, the sizing and the commands to run, without deploying anything. Memory is the ONNX files times two plus 300 MB for the runtime; CPU is the target requests per second times the measured CPU latency at 60% utilisation; both round up to the smallest valid Fargate pair. The stack is one Fargate service behind a load balancer: the task's file system is read-only, only the load balancer can reach it, it scales on CPU, and its execution role may read exactly one Secrets Manager secret, the bearer token the runtime now checks on every request except `/health`. No credential is written to any file: the AWS CLI uses the person's own sign-in, and the token is created straight into Secrets Manager. The template passes `cfn-lint`. The listener is HTTP; an HTTPS listener with a certificate is left to the person, because it needs their domain.

## A test set is never empty

A data build can end with every row in train: LLM-written rows that stopped early, or a source read only as positives. Nothing could then be scored, and every later stage failed in its own way. Now, when dev or test is empty and train has at least 50 rows, a share is held out per label with near-duplicates kept on the same side (the same split used for Hub datasets), and the log says how many rows moved. If a build still cannot be scored, the autopilot stops with that reason.

## Text inputs are read in full

A spec's `train.max_len` defaulted to 128 tokens and nothing checked it against the data, while tokenisers cut from the right. For inputs that end with what matters (a conversation whose last turn decides the answer) the model never saw the deciding part: tool-call-gate's median input was 145 tokens and its 90th percentile 382. Each build now counts the training rows' tokens with the backbone's tokeniser and, when the 95th percentile is longer, raises `max_len` to cover it (rounded up to 64, at most 1024 or the model's own limit) and saves it in the experiment's spec, so export reads the same length.

## LLM-written rows are written once

A data folder is keyed by everything in the data plan, including the Hugging Face sources, so adding a source started a new folder and the LLM wrote every row again; rows written for an experiment that never finished were lost to later runs. LLM rows depend only on the labels, the description and the synthesis fields, so they now carry that narrower key (`synth.key` in the folder), and a build takes rows already written under the same key in the specialist's other data folders, per split, before asking the LLM for the rest. Positive and negative sources were only ever mixed into audio; for other modalities they now feed the unlabeled pool (pretraining and teacher labelling) instead of being read and dropped.

## Choosing among options offered with the input

Tool calling has no fixed label set: each request comes with its own tools, and new tools appear after training. `byosm/options` turns such data into (situation, option) pairs labelled fits or no_fit, so a specialist scores one option at a time and any option, seen or not, can be chosen; the decision is every option at or above a threshold, or none. A share of situations is paired with another request's unrelated tools so that choosing none is learned. Whole situations go to test, and to dev, so no request is trained on and tested, nor trained on and used to pick the epoch; a local table can mark rows `dev` as well as `test`. Each pair also lists the names of the other options offered with it, so a pair model can tell the better of two related tools while still scoring one option at a time. The decision is scored per situation, exactly the right set or not, with separate accuracy for requests that need a tool and those that need none.

## Models that embed an encoder

Some decision models ship an encoder inside a custom architecture (laya: a ModernBERT-large encoder plus its own decision heads in one safetensors file). `byosm/encoders.py` extracts the encoder once into the cache as a plain Hugging Face folder (its config, tokenizer and the prefix-stripped weights), and the text modality resolves the backbone name to that folder, so such a model can be fine-tuned like any other.

## GPU share follows what is free

The Apple GPU used to be split into fixed shares (90% divided by the slot count, 45% with two slots) so two trainings could always run side by side; a training alone then used half the GPU. Now each slot records the share its training took (`slot-N.share` next to the lock, because opening a lock file empties it), and a new training takes whatever is left of the 90% ceiling, waiting while less than a quarter is free. A training alone gets the full 90%, about 47 GB on a 64 GB Mac, and a second one waits until it gives the share back. Shares are reserved at start rather than measured, because GPU memory is allocated lazily and two trainings that each started small could otherwise grow past the machine together. The share reaches the child process that runs each attempt through an environment variable.

## Pretraining reads shorter text

LeJEPA pretraining passes six views of every input through the encoder with gradients (two masked copies at full length and four crops), so its memory is six times that of a fine-tuning batch and grows with the square of the length. When fine-tuning reads more than 256 tokens, pretraining now reads at most 256: it only adapts the encoder to the domain's language and does not need the whole input. If memory still runs out at the smallest batch, the retry trains without pretraining instead of failing the job.

## Continuing a trained model

`byosm continue <name> [--from v4] [--data new.csv]` makes an experiment whose `train.init_from` names its parent. Training then loads the parent's weights instead of the raw backbone: every tensor of the same shape is kept, and for a head whose labels changed each label keeps its row and new labels start fresh. Pretraining is skipped because the encoder is already adapted. The recipe is gentler, half the parent's encoder learning rate and at most three epochs, and when the data changed, half as many rows again are replayed from the parent's training set (never rows in the new dev or test), so the model learns the new rows without forgetting the old ones. A new table is checked like `check-data` before anything trains. The continued experiment is selected against its parent on calib like any other.

Without options, `byosm continue` (and the Keep training button) plans the step from the promoted model's health checks and its dev errors: overfitting gives one epoch at a quarter of the learning rate; a model behind its target or rival whose dev score was still rising at its last epoch gets more epochs; thin data raises how many rows the LLM writes (300, 600 or 1000), aimed at the dev errors with the weakest label's errors first, and varied in phrasing when the model drops on new inputs. Checks that training cannot fix, an uncertain test score or a slow model, are named with what to do instead (a bigger test set, distillation). The button shows the plan before it runs, and the job log records every decision.

## Jobs and a sleeping Mac

When the Mac sleeps, a running job freezes; on waking, the stall watchdog used to see hours without output and killed healthy trainings (two at once, after a night). The watchdog now notices a clock jump much larger than its polling interval and does not count that time as silence. While a job runs, `caffeinate -i -w <pid>` also keeps the Mac from idle sleep, ending by itself with the job; closing the lid still sleeps the machine. `keep_awake: false` in `byosm.yaml` turns that off.

## Time series against Timestral

`byosm series run <domain>` trains a forecasting specialist on a GIFT-Eval domain (energy, sales, operations, or any single dataset) and scores it with the rivals by the same code. The protocol follows GIFT-Eval's short term: the horizon comes from the frequency, the last windows of each series are the test (up to 20, about a tenth of the shortest series), and specialists train only on values before the first test window. MASE is scaled by each series' seasonal naive error, CRPS comes from nine quantiles, and a domain is summarised by the geometric mean of each dataset's score relative to seasonal naive, as GIFT-Eval does. Every model's training seconds and milliseconds per forecast are recorded. Timestral is scored through its own package when it is installed; the specialists are a gradient-boosted global model (a squared-error median on scaled lags, a seasonal lag, the step and recent level statistics, with quantiles from its own errors on the last tenth of the training origins; an absolute-error loss collapses to zero when half the targets are zero, as solar at night) and a small patch transformer (about 0.6M parameters: 512 steps of context in patches of 16, three layers, all nine quantiles for the whole horizon in one pass, trained with the quantile loss on random windows of the training history).

## GPU queue

Each training writes its share, its job id and when it got the GPU next to the slot lock, so `/api/gpu` can list who
holds the GPU and how much of it. A running job whose last event is a GPU wait is in the queue. The Jobs page shows
both above the timeline, so a long wait is visible without opening job logs.

## Interval width per series

The boosted forecaster learns one set of error quantiles, in units of each series' level. On a dataset that mixes
smooth high-volume series with noisy small ones, that width fits neither. The errors are now divided by each series'
own seasonal-difference noise before the quantiles are taken, and multiplied back at forecast time. CRPS on
electricity went from 0.198 to 0.094 (seasonal naive 0.102); the median forecast, and so MASE, is unchanged.

## CPU fallback for small trainings

A training can ask `gpu_slot` for a fallback device. When the GPU has no room at that moment, it runs there at once
instead of queueing. The time series patch transformer (0.6M parameters) uses it while its training is at most
`CPU_FALLBACK_STEPS` (12,000) steps: that trains on the CPU in about 45 minutes, while a queue behind a large training
can last hours. Keep training doubles the steps each round, and past that limit the CPU loses: an energy run at 192,000
steps went at 500 steps per 110 s on the CPU, about 12 hours, against about 3 hours on the GPU after a wait of minutes.

## Converged boosting is not trained again

`byosm/series/models.py` `Boosted.fit`: Keep training doubles boosting's `max_iter` each round, but on these tables
scikit-learn stops early once the validation score stops improving, so a larger `max_iter` with the same data and
seed gives the same model. Each fit is cached under a key of its training table, with the iterations it used; a later
round with at least as many allowed iterations reuses a fit that stopped before its limit, and a retried round with
the same limit reuses its fit outright (the patch transformer resumes from its checkpoint), so an interrupted Keep
training does not repeat hours of finished work. Energy's boosting took
8,787 s and then 11,434 s for rounds whose extra iterations were not used. Not yet measured on a real run: the next
Keep training of a domain shows the time saved.

## Boosting predicts a correction to seasonal naive

The boosted forecaster predicts how far each step will be from the seasonal naive value, in units of that series'
own noise, capped at ten units. A series that barely moves (some GIFT-Eval service metrics sit at 0.74 with a 0.001
step) then gets seasonal naive plus a small correction, instead of the model's usual error on a level scale. The cap
matters: without it, one step change on a constant stretch becomes a target of hundreds of units and dominates the
squared-error fit.

## Published rival scores

Timestral's GIFT-Eval scores are read from the leaderboard's per-dataset results (cached by URL), not re-run here.
They count only where our seasonal naive MASE is within 2% of the leaderboard's seasonal naive on the same dataset,
which shows the split and the metric match. A domain's overall includes a model only if it has a score on every
dataset of that domain.

## Forecasters are specialists

A GIFT-Eval domain (energy, sales, operations) is a specialist of kind `forecast`, listed with the others once it has
been run and shown on the same specialist route with a page of its own: rivals board, versions, datasets. Like teams it
lives outside `TaskSpec`, whose labels and classifier steps do not apply. Each run fits every recipe once on the history
before the calib windows, picks the recipe with the lowest calib MASE, scores test with that same fit (later history
is only input), and adds a version with a 95% bootstrap interval over test windows. A version is promoted only if its
calib MASE beats the promoted one. Lower is better, so the lead over a rival is the rival's score minus ours.

## Keep training until a rival is beaten

A forecaster's next version trains with twice the steps of the last (patch transformer steps and boosting
iterations, `2**effort` where effort is one more than the promoted version's). `series run DOMAIN --until-rival`, which is
what Keep training starts, adds versions until the promoted one beats the best rival or `--rounds` is spent.
Promotion still decides on calib. The stop condition reads test, because the published rival scores exist only on
test; stopping on the first win leans the reported number slightly optimistic, so every version stays on the page
with its interval.

## One model per forecaster

A forecaster's version is one model per recipe, trained on every dataset of its domain together (`series/fit.py`),
not one model per dataset. Datasets in a domain can differ in season and horizon (sales mixes daily and monthly), so
models fit on groups that each carry their own season and horizon: boosting takes the season as a feature and a
fixed lag window, and the patch transformer trains each series with its own horizon from one 60-step head. Scores are
still reported per dataset, and the domain score is the geometric mean across them.

## Event risk is a specialist kind

The "will this entity end within N years" models built from open data registers are specialists of kind `risk`,
listed with the others and opened on the same route. Their score is gradient boosting's test AUC and their rival is
the better of logistic regression and the base rate, both already scored by the same code. They are only created
from a sentence: `auto` routes it, finds the table on open data portals and designs the spec, so there is no form.

## Rivals are compared on shared datasets

Published rival scores exist only where the protocol check passes, so a rival can cover fewer datasets than the
forecaster. Each rival is compared with the promoted version on the datasets both were scored on (geometric mean of
MASE relative to seasonal naive), and the hardest one, with the smallest margin, is the card's rival and the target of
Keep training. Each version keeps its per-dataset scores, so the comparison does not change while a new round is half
done. GIFT-Eval calls `car_parts_with_missing` just `car_parts`.

## Give GPU memory back

PyTorch's MPS allocator keeps every block it has used and never returns it unless asked. A text encoder at batch
size 8 was holding 33 GB of GPU memory, leaving 5.7 GB free on a 64 GB machine and every other training queued. The
cache is now released after each epoch's dev evaluation and after each time series training (`release_cache`).

## Each dataset gets an equal share of boosting rows

With one boosted model per domain, rows were budgeted per series, so electricity's 370 series supplied almost all of
the 1.5M training rows and the 7-series ETT datasets about 1% each. The row budget is now split equally between the
domain's datasets, then between each dataset's series.

## Pretrain a base forecaster, then specialise it

Timestral-5M, zero-shot, beats every forecaster we trained from scratch on each domain's own history, so the gap is
pretraining data, not size. `series corpus` downloads a balanced sample of GIFT-Eval's pretraining corpus (152
datasets built not to overlap its test sets; 50 GB by default, every dataset capped at an equal share). `series
pretrain` trains a 6M-parameter base patch transformer (1024-point context in 32-point patches) on windows drawn from
those memory-mapped files, 80%, and from the training portions of the GIFT-Eval datasets, 20%, reporting progress
every 500 steps. Each forecaster then fine-tunes that base as its "pretrained" recipe, chosen on calib like the others.
The small from-scratch model (2.8M parameters, 512-point context) stays as the recipe that trains on a CPU in minutes.

## Ready, and ready for production

A specialist is ready once it has a scored version. It is ready for production when its health score is at least 80
with no blocking check failed or unknown. Beating the best rival is one of the health checks and counts in the score,
but it is advisory: an owner may require it, the app does not.

`next --all` runs the same loop as `byosm ready` (`byosm/auto/ready.py`): it reads the health, runs the untried step
with the largest gain, reads the health again and logs whether the checks that step targets moved. It used to compute
the step list once and run every step in a fixed order, so a step ran even when an earlier one had already fixed, or
could not fix, the check it was meant for. A step that clears a production blocker goes before one that only moves
an advisory check: tool-choice's run picked "Find more experiments" three rounds in a row, credited with the advisory
rival check, while overfitting and robustness, the two blockers, waited.

Every failing check has a step. A failing "Holds on new inputs" now offers Keep training (`continue --run`, titled
"Train on varied inputs"), whose plan writes rows that vary phrasing and context; before, the check had a step only
while a second suite was unscored, so tool-choice's -0.233 on llm-written was a blocker with nothing to run. "Make it
faster" is not offered again once the promoted model's student (`distill.student_name`) has been built.

## Size against the rival is part of every result

Accuracy alone made a 109M-parameter model built on its rival's own checkpoint look like a win over that rival. Health
now has a "Smaller than the rival" check (4x smaller or more is ok, smaller is a warning, bigger fails; advisory like
the rival check), and the model card's facts carry `size_vs_best_rival`: how many times smaller the model is and
whether it is built on the rival's checkpoint. The card's headline must state both.

## Distill by default

The autopilot distills its promoted model whenever it is above the size target, whether or not it beat its rival,
and ships the student when its calib score is within the spec's `speed_tolerance` (0.01 by default) of the teacher's. A specialist that
is not at least 4x smaller than its rival gets a "Distill into a smaller model" next step from its health.

Any build is promoted for speed by the same rule (`byosm/pipeline.py` `fixes_speed`, the same `speed_tolerance`): it
meets the latency budget that the promoted model misses and its calib score is within the tolerance. Before, only the
autopilot applied the tolerance; a "Make it faster" step built the student through `maybe_promote`, which asked for a
sure calib win, so the step could never pass the speed check it was chosen for. tool-choice's 11M student (10 ms,
calib 0.956) stayed out at the default: the 163 ms teacher's calib is 0.978, 0.022 ahead. The tolerance is the
spec's `speed_tolerance`, how much calib a specialist gives up for meeting its latency budget; tool-choice sets 0.025,
since its student keeps 97.4% of fits caught at 16x the speed and 1/35 of the size.

## A rival score must be a fair bar

`byosm/health.py` `_rival` marks the rival check unknown, with the reason, in two cases where the number cannot be
trusted. When the rival is an LLM and the builtin test set was written by a model of the same family
(`model_refs.family`, the first word of the model id): tool-call-gate's 834 test rows were written by Claude Opus and
kept only where Claude Haiku agreed, and the rival is Claude Haiku at 0.98. And when the rival misses the metric's
constraint, so it scores exactly 0 (`metrics.unreached`): on turn-completion every rival, including
livekit-turn-multilingual at 0.944 precision against the 0.98 required, scored 0, which the check read as a sure
+0.895 lead.

The same rule marks written test sets (`byosm/evals/test_sets.py` `same_family`): each set's rows record the model that
wrote them, and an LLM rival of that family gets an info icon beside its score. On tool-choice, Haiku wrote one set
and scored 27 of 27 on it.

## The strongest rival, measured

Fine-tuned rivals are searched by the specialist's own task (text, token, audio and image classification), and up to
three are added per benchmark. Candidates are ordered by the score they report for the benchmark in their Hugging Face
metadata, then by size, since the larger fine-tuned checkpoint is usually the stronger one. Rivals already on the
board and models published by BYOSM are skipped. Every added rival is then scored by our own code, so a self-reported
number never reaches a card; the board's best measured rival is the one cards and health compare against.

## Files shared with running jobs keep their format

Running jobs keep the code they started with, so files several jobs read must be readable by every version still
running. The GPU share file changed from a number to JSON and back, and each change crashed jobs started under the
other version. The share file is a number again, the holder's job and start time live next to it in `slot-N.holder`,
and the reader accepts both a number and the old JSON, because a job writing JSON can still be alive.
## Keep training stops at a plateau

Doubling a forecaster's training each round gained under 1% of calib MASE per round on all three domains (energy
0.849, 0.842, 0.838 on test) while doubling the CPU time. Keep training now stops when a round improves the best calib
MASE by less than `MIN_GAIN` (1%) and names the next lever: the pretrained base model, or a bigger one once the base
exists. The pretrained recipe joins every round automatically when its weights are on disk.

The stop is remembered (`plateau` in the series report, with the effort and the pretrained base and corpus size at the
time). The next Keep training does not double again while those are unchanged: it pretrains the missing base size
first, or, when every size exists, trains nothing and names more pretraining data. Before, effort was the number of
versions, unpromoted ones included, so every click doubled the steps after the plateau: sales went to 96,000 steps
and energy to 192,000 with no calib gain. The rival is also compared only with the promoted version's own scores;
when those were missing it fell back to the latest run's models, which had compared a different model with the rival.

## Admit trainings by the memory that is really free

Reservations alone kept three trainings waiting for hours while the GPU holder used 2.3 GB and the machine had 35 GB
free: the first training reserves up to 90% of the GPU so it can grow alone, and nothing else fit the 10% left. On
Apple silicon a waiting training's share is now the larger of what the reservations leave and the share of memory
actually free, so it starts when real memory allows. If two trainings later grow into each other, the out-of-memory
recovery halves the batch.

## A silent job is stuck only when it is also idle

The watchdog stopped three forecasters that were training on the CPU for over 90 minutes without printing. A job
that is silent past the limit is now stopped only if it also used less than a minute of CPU (itself and its children)
in that time; a busy job resets the clock. Long trainings also report their step every 500 steps, so they show
progress and time left instead of going quiet.

## Forecasters have a health score

A forecaster's health uses the same scoring as a classifier's (`health.report`: weighted checks, production ready at
80 with no blocking check failed or unknown) with checks that fit forecasting: it beats seasonal naive on every
dataset, its calib and test scores agree, its test interval is narrow, and, advisory, it beats the hardest rival on
the datasets both were scored on. It is served by the same `/api/health/{name}` and shown on its page.
Calib and test must agree on every dataset, each relative to seasonal naive, and the check names the worst one: the
domain geomean showed energy 4% apart while solar went from 0.82 of naive on calib to 1.10 on test, 34% apart.

## The GPU goes to whoever waited longest

When the GPU freed up, whichever waiting job polled first took it, so a pretraining job waited four and a half hours
through three handoffs. Each waiting job now leaves a ticket with its start time in `.byosm/gpu/queue`, and only the
oldest live ticket may take a free slot. Tickets of jobs that died are removed. Small trainings with a CPU fallback
never queue.

## Robustness suggestions that can be acted on

The "Holds on new inputs" check suggested `eval` for suites whose test rows the promoted model had trained on (folded
in), which `eval` rightly never scores, and `testset` for tagging models, which it cannot write tests for. It now
skips folded suites, suggests writing a fresh test set only for classification, and says so otherwise. The scoring
primitives that classifier and forecaster health share live in `health_score.py`.

## An LLM-written test set is judged against fine-tuned rivals

Independent test sets are written by an LLM, so they follow the LLM's reading of the label definitions. Every model
fine-tuned on a benchmark drops on them (emotion: ours 0.898 to 0.536, j-hartmann distilroberta 0.838 to 0.438), while
the LLM rivals score 0.95 to 1.00 because the set is in their own style. An absolute drop therefore measured how far a
benchmark's labelling sits from an LLM's, not whether a model generalises. "Holds on new inputs" now compares the
model's score on its hardest other suite with the best fine-tuned rival on that same suite (ok within 0.05, warning
within 0.15), and keeps the absolute drop only when no such rival was scored there.

## Each backbone remembers how much GPU memory it needs

Admitting a waiting training by free memory let one start on a 40% share that it then outgrew (laya, about 33 GB, out
of memory at batch 8 with nothing smaller to fall back to). The peak memory a training holds is sampled each time the
cache is released and saved per backbone when it leaves its slot (`.byosm/gpu/needs.json`); a run that ran out of
memory records 25% more than the share it had. A backbone with a recorded need is admitted only once that much is
free. All memory accounting (free, total, needed, cache release, needs) lives in `train/gpu_memory.py`, which the job
admission in `resources.py` also uses instead of its own copy of `free_gb`.

## Long inputs keep their head and their tail

The tool-call gate's own error analysis found that every dev example with 15 to 25 tools was wrong: the tool list comes
first, the deciding user turn last, and plain truncation at the length cap cut the user turn. A text classifier can
now read long inputs head and tail (`train.truncation: head_tail`): the first quarter of the token budget from the
start and the rest from the end, the split found best for long-text classification with BERT-style models. New text
specs use it; older models keep `right`, and the setting travels in the bundle's manifest so the one-file runtime
(`rt.head_tail`, which training imports too) reads inputs exactly as training did.

## Every job runs a frozen copy of the code

Python imports modules when first used, so a job that runs for hours loads some modules at the version it started
with and others at whatever is on disk later. The tool-call gate crashed that way: its old `TrainCfg` met today's
`text.py`. Each job now runs from a snapshot of the `byosm` package taken when it starts (`app/snapshot.py`):
the source is hashed (the hash is cached until a file changes), copied once per version into the machine cache under
`code/<hash>`, put first on the job's `PYTHONPATH`, and the job starts with `python -P` so the working directory
cannot shadow it. Snapshots older than a week are removed.

## A collapsed training is a diverged one

laya-encoder's refine round fell from dev macro F1 0.534 to 0.182 in one epoch, the score of predicting one class,
while training in bfloat16. Only NaN outputs counted as divergence, so it would have trained six more epochs on
nothing. Once a run has reached macro F1 0.4, a dev score below half its best raises `DivergedError`, and recovery
retries at the next precision (bfloat16, then fp32) as it does for NaN.

## Robust scaling for the patch transformer

Each input window is scaled by its own mean and standard deviation. On the operations data, nearly flat 10-second
windows gave a standard deviation close to zero, so the next jump became a normalised target in the hundreds and
single training batches hit losses of 202 and 72 against a usual 1. The scale now has a floor of 1% of the window's
mean level and training targets are capped at 50 scaled units; ordinary windows are unchanged.

## Only the small forecaster falls back to the CPU

A training that cannot get the GPU falls back to the CPU once, at its start, and stays there. That suits the small
patch transformer (2.8M parameters, 512-point context), but the pretrained base (6M, 1024-point context) fine-tuned at
about 1,000 steps an hour on the CPU while the GPU freed up beside it. The pretrained base now waits for the GPU.

## The GPU queue backfills

With strict first come first served, a job that needed 34 GB held the head of the queue while 29 GB were free, so a
job that fitted waited behind it and a slot sat idle. Each ticket now records the share its job needs, and an older
ticket blocks a newer one only if the older job could start with the share free now; otherwise a job that fits goes
first.

## A GPU slot is released under the admission lock

A finishing training deleted its share file before it unlocked its slot. A training that looked in between saw a held
slot with no share and assumed the default half, so it started with half the GPU when the whole of it was about to be
free. Releasing takes the same lock as admitting, so a waiting training sees a slot either fully held or fully free.

## A cached score belongs to one training of an experiment

Scores were cached by test set only, so when the autopilot retrained `v1` in place, the board kept the old `v1`
score and the keep/revert decision compared against a model that no longer existed. Each experiment's cached score now
carries a hash of its training metadata, which changes on every training, so a retrained experiment is scored again.
Scores cached before this change are adopted once, like baseline scores were.

## Every forecaster version can be run live

A forecaster used to be scored and then thrown away, so there was nothing to forecast with. `byosm/series/run.py`
`run_domain` now saves the model behind each new version's recipe to `.byosm/forecasters/<domain>/<version>.joblib`
(`byosm/series/saved.py`); an ensemble is saved as the trained models it averages (`Ensemble` in
`byosm/series/models.py`, also used to score it). The Try it tab forecasts a random test window of any of the domain's
datasets with the promoted version, next to what really happened, or a series pasted by the user. Versions built before
this change have no saved model until the next Keep training.

## The board shows rivals scored on fewer datasets

The board scored only models present on every dataset, so a rival published for two of three datasets disappeared from
it. Each model is now scored on the datasets it has (`datasets` in `overall`), and a rival with fewer shows our
promoted model's score on the same ones (`ours_on_shared`), as the rival stat card already did.

## Calib is held out of dev on builtin suites

On a builtin suite the calib split used to be the dev split, the same rows that picked each model's best epoch and its
threshold, so a promotion vote rewarded whichever run happened to overfit dev best. Every dataset now keeps a fourth
split, `calib`, carved from dev by `carve_calib` in `byosm/data/splits.py`: rows are grouped so near-duplicates and
clips of one recording stay on one side, ordered by a hash of their id within each label, and every second group goes
to calib. Training still selects on dev; selection, promotion and the held-out size checks read calib.

Each training records `calib_held` in its metadata. A model trained before the split saw today's calib rows while it
picked its epoch, so it can no longer win a promotion on a builtin suite (`clean_selection` in
`byosm/evals/selection.py`); it stays promoted until a model trained on the new dev beats it. The existing datasets
were split once, and builtin scores carry the split in their fingerprint so every model is scored again on calib.

## A lead must beat seed noise before it wins

A single training per experiment let a lucky seed win promotion and an unlucky one bury a good idea. When a new model
leads the promoted one on calib, `maybe_promote` (`byosm/promotion.py`) now trains both recipes again on other seeds
until each has `finalist_seeds` (3 by default, in `spec.yaml`) runs whose calib was held out. Replicas are named
`<recipe>-s<k>` (`byosm/replicas.py`) and never promote themselves. The vote then uses the mean calib score over seeds,
a paired bootstrap that resamples the same calib rows for every seed of both sides (`bootstrap_runs` in
`byosm/resample.py`), and a seed-noise bar: the lead must exceed 1.96 pooled seed standard deviations times
sqrt(1/n_a + 1/n_b). The pooled standard deviation comes from every recipe with more than one clean run, so it is
measured once per specialist and reused.

The health page shows the noise as "Seed noise measured", a warning rather than a blocker while the promoted model has
one seed, and `byosm seeds <name>` trains the missing seeds.

## A repeat is caught by its normalised edit

"Already tried" compared the exact JSON of a proposal's edits, so a learning rate of 2.2e-5 after 2e-5, or the same
two edits in another order, counted as a new idea. `proposal_key` (`byosm/improve/proposals.py`) now sorts the edits,
folds text to lower case with single spaces, and rounds every number to the nearest quarter decade on a log scale,
so values in one bin (a factor of 1.78 wide) share a key. Seeds keep their exact value, because a re-seed is a
deliberate repeat.

## Every experiment records its GPU time

To share trials out by gain per GPU-hour, the cost of each experiment has to be known. `gpu_slot`
(`byosm/train/gpu.py`) adds up how long a process holds a GPU slot, and `build` stores the difference over the build
as `gpu_hours` in the experiment's metadata; a training on the CPU records zero. The Runs tab shows it next to the
training time.

## The proposer picks from a technique catalogue

The proposer could only edit a flat list of spec paths, and its prompt told it to prefer data fixes everywhere, so its
proposals fell into about six families and repeated. `byosm/improve/techniques.yaml` now lists every technique family
with its editable paths, the kinds it fits, a relative cost in trainings, a prior, the evidence behind it and whether it
is available or still planned. The editable paths are the union of the available families' paths
(`allowed_paths` in `byosm/improve/techniques.py`), so a family that is not built yet cannot be proposed. Every
proposal and every past experiment is tagged with the family of the paths it edits (`family`), and the LLM reads the
families that fit the specialist with how often each was tried, and is asked to spread its proposals over them.
`train.seed` left the editable paths: seeds are trained by the replicas, not proposed.

Planned families (an embedding-model head, SetFit, label-noise cleaning, weight averaging, learning curves,
forecaster pretraining v2, re-initialised layers, a small search, R-Drop and mixup, quantile boosting, self-training)
become available as each one lands in the pipeline. Multi-rate forecasting and routing are available to forecasters
through Keep training (below). Input layouts (`data.pairs.layouts`, `mask_names`, `paraphrases`) and near-miss
negatives (`data.pairs.near_misses`) are available. A family can also list `requires`, spec paths that must be set for
it to fit (`eligible`): both pair families require `data.pairs`, so they are offered only to a specialist made by
`byosm options new`, whose pair table they rewrite. The held-out fields (`held_out_apis`, `held_out_layouts`) are in no
family, because they decide calib and test.

Planned families (input layouts, near-miss negatives, an embedding-model head, SetFit, label-noise cleaning, weight
averaging, learning curves, re-initialised layers, a small search, R-Drop and mixup, self-training) become available
as each one lands in the pipeline. Multi-rate forecasting, routing, quantile boosting and forecaster pretraining v2 are
available to forecasters through Keep training (below).

## A bandit shares the trials out over technique families

The autopilot picked proposals from a fixed table of priors keyed by rule name, so the same rules won every round.
Proposing is now a search over the families of `byosm/improve/techniques.yaml`:

| Piece | Where | What it does |
|---|---|---|
| Verdicts | `byosm/promotion.py` `_record` | every promotion vote stores `verdict:<exp>`: won, P(better), seeds, family, GPU hours |
| Bandit | `byosm/improve/bandit.py` | Thompson sampling over families: Beta(1 + prior x 2 + wins, 1 + (1 - prior) x 2 + losses), wins and tries of other specialists of the same kind counted at half weight, the draw divided by the family's cost |
| Round one | `family_order` | families never tried on this specialist (prior at least 0.3) come first |
| Dropped branch | `_dropped` | a family with three tries and no win on this specialist is skipped while others remain |
| Rules as priors | `RULE_BOOST` | a family a diagnosis rule proposed is drawn with 1.5 times its score, instead of the rule deciding |
| Portfolio | `byosm/improve/portfolio.py` | up to three transferable edits (backbone, schedule, head, pretraining, regularisation, precision) that won on other specialists of the same kind are proposed first |
| Candidates | `byosm/improve/analyst.py` | the LLM writes 5 to 10 proposals, each with a diagnosis, a hypothesis and its probability of a sure lead, at least two of them unlikely ones |
| Novelty | `_novel` in `proposals.py` | an LLM idea whose text is a near copy (5-word shingles, 0.9) of an earlier experiment's is dropped, as is any proposal that leaves the spec unchanged |
| Order | `rank` | one proposal per family in bandit order, the most probable first, before a second from any family |

The Runs tab shows the search: families tried, sure leads, GPU time per sure lead, repeated edits, and tries against
sure leads per family. Experiments from before verdicts were recorded count as a win only if they are the promoted
model. Numeric values inside a family come from the proposer, and from the TPE sampler described below once a setting
has been tried enough.

## Label noise is cleaned by out-of-fold loss and the judge

LLM-written and panel-labelled training rows carry some wrong labels, and the model learns them. With
`data.clean_labels: true` (off by default, a `label-noise` family the proposer can choose), the data build scores
every training row with a cheap character n-gram TF-IDF and logistic-regression model trained on the other folds
(`byosm/data/noise.py`, five folds), sends the 10% with the highest out-of-fold loss (at most 2,000) to the judge model,
and drops the ones the judge labels differently. Only train is cleaned: calib and test keep their rows so the clean-up
cannot flatter the score. It has not been measured on a specialist yet.

## A winning recipe ships as a seed soup

Once a recipe wins the seed-averaged vote, its seed runs (at least two whose calib was held out) are averaged weight by
weight into `<recipe>-soup` (`byosm/soup.py`): floating-point tensors are averaged, integer buffers are copied from the
first run. The soup is scored like any experiment and ships instead of the single run when its calib score is at least
the mean of the seeds; otherwise the original run ships. A soup maps back to its recipe, so a later challenger is
compared with the recipe's seed runs, and it never counts as a seed when the noise is measured. Seed runs share their
pretrained start, which is what makes their weights averageable. The catalogue marks it `built-in`: the pipeline
always does it, so it is not a proposal. Not measured on a specialist yet.

## Candidates are screened with short runs first

Every candidate used to cost a full training, so a round could try one idea. Batches now go through one rung of
successive halving (`byosm/rungs.py`): each candidate trains as `<exp>-r1` for 1/`screen_eta` of its epochs (3 by
default, in `spec.yaml`) with no refine rounds, never votes on promotion, and only the best ceil(n/`screen_eta`) on calib
go on to a full training and the seed-averaged vote. `byosm improve --each --run` (the Run experiments step) screens
the lined-up batch, and the autopilot screens the first `screen_eta` proposals of each round and trains the best one.
A candidate dropped at the rung keeps its attempt, so it counts as a try without a win for the bandit and is not
proposed again. Rung runs and ablation arms are `auxiliary` runs, hidden from the Runs tab. A rung is a planned
budget, not a cancellation: every short run finishes.

## Ettin encoders and pair checkpoints join the backbones

Encoders beat same-size decoders on classification, and the Ettin encoders (17M, 32M, 68M, MIT licence) are
ModernBERT-style models trained on the same recipe at every size, so they fill the gaps in the escalation ladder: from
MiniLM the next rungs are now Ettin-32M, DistilBERT, Ettin-68M and DeBERTa-v3-xsmall. Two checkpoints trained on pairs
of texts, the MS MARCO MiniLM reranker (the tool-choice backbone) and an NLI DeBERTa-v3-xsmall, are listed with
`"for": "pairs"`: the proposer can pick them for a pair task, and the escalation ladder skips them for single-text
tasks. None of the three Ettin sizes has been trained by BYOSM yet.

## Forecasts mixed across sampling rates

A small forecaster sees a long series only through its last few hundred steps. `MultiRate`
(`byosm/series/multirate.py`) wraps any trained forecaster and also forecasts the series at coarser rates, with no
training: blocks of r steps are averaged so the last block ends at the forecast origin, the season and horizon shrink
by r, and each coarse quantile is interpolated from the block centres back to every native step. The rates are
`MULTI_RATES` (1, 2 and 4) in `byosm/series/prediction.py`. Each series scores its rates per quantile by the pinball
loss on its own last `BACKTESTS` (2) horizons before the origin, so the weights never see a calib or test value, and
weighs them by a softmax of each loss relative to the best one (`SHARPNESS` 10): a rate at 1.5 times the best loss
gets under 1% of the weight. The mixed quantiles are sorted. A series too short for the coarsest rate keeps the native
forecast. The cost is rates times (backtests + 1) forecasts, 9 by default.

Inverse-loss weights were too flat. Boosting reads coarse lags as if they were native ones, so its coarse forecasts
are poor, and on 20 synthetic hourly series (a sanity check, not a benchmark) inverse-loss weights took its test MASE
from 0.858 to 0.997. With the softmax the twin scores 0.870, and seasonal naive still gains, from 1.272 to 1.222. The
twins took 7 to 8 times as long per forecast there.

It is a recipe option of a version, `predict.rates` (default `[1]`, off). When it is on, each of our models and their
ensemble get a twin named `<model>, multi-rate`, scored by the same metric code on the same calib and test windows,
and the calib pick decides whether a twin serves. The next training starts from the promoted version's options, so a
twin that wins stays on. Two gotchas: twice the candidates leans the calib pick slightly optimistic, and a trained
model has seen the backtest windows in training, which can favour the native rate. TimeSSM reports 0.545 to 0.513
CRPS from this on a 2.5M model (self-reported); it has not been measured on our forecasters yet.

## A model per dataset, picked on calib

One recipe per domain cannot use the patch model on one dataset, boosting on another and seasonal naive where nothing
beats it. With `predict.routing` on (default off), `route` in `byosm/series/routing.py` picks, for each dataset
configuration (GIFT-Eval's dataset and frequency; every domain here is short term), the candidate with the lowest calib
MASE among our models, their ensemble, their multi-rate twins and seasonal naive, and the pick joins the board as the
candidate `routed per dataset`. Its calib score is cross-fitted: the calib windows are split into `FOLDS` (2)
interleaved halves and each half is scored with the model picked on the other, because the plain per-dataset minimum
would always win the calib pick. Its test score uses the pick made on every calib window.

Each dataset's pick is stored as `route` in its routed row. The board (`promoted_board`) and the specialist card
(`summary`) list them as `routes`, the Datasets tab shows them, and the saved version (`Routed`) serves a live forecast
with the model of the dataset it names, or with the model most datasets were routed to for a pasted series. A dataset
routed to seasonal naive counts as not beating seasonal naive on the health check. GIFT-Eval entries that combine or
route models gain 8% to 12% CRPS over their base model; it has not been measured on our forecasters yet.

## Keep training tries prediction-time techniques first

Forecasters live outside `TaskSpec`, so the proposer never reaches them; their search is Keep training
(`until_rival` in `byosm/series/run.py`). Each round now first asks `next_technique` (`byosm/series/prediction.py`)
for the catalogue families that fit the `series` modality and whose paths name a field of the version's prediction
options (`predict.rates` for `multi-rate-forecast`, `predict.routing` for `forecast-routing`), best prior per cost
first. A family is tried when turning its option on over the promoted version's options gives options that no version
at the promoted effort has had. That round loads the promoted version's saved models (`members` in
`byosm/series/combine.py` unwraps ensembles, twins and routed models), scores them again on the same calib and test
windows with the new options (`rescore` in `byosm/series/fit.py`, no training), and adds a version at the same effort
that is promoted only if its calib MASE is better. Rival rows the round does not score again are kept from the parent
version, and so are the models' training times.

Only when every family has been tried at the promoted effort does a round train, with twice the steps and the
promoted version's options, so a technique that won stays on and one that lost is tried again on the next promoted
models. Technique rounds never count towards the plateau stop and keep a stored plateau, and they run even when the
plateau would stop training, since they cost no training. A promoted version without a saved model trains instead.
`series run DOMAIN --until-rival --rounds 2` applies both techniques to the promoted models; neither has been run on
the GIFT-Eval forecasters yet.

Planned families (multi-rate forecasting, routing, input layouts, near-miss negatives, an embedding-model head,
SetFit, label-noise cleaning, weight averaging, learning curves, forecaster pretraining v2, re-initialised layers, a
small search, R-Drop and mixup, quantile boosting, self-training) become available as each one lands in the pipeline.

Planned families (multi-rate forecasting, routing, input layouts, near-miss negatives, label-noise cleaning, weight
averaging, learning curves, forecaster pretraining v2, re-initialised layers, a small search, R-Drop and mixup,
quantile boosting, self-training) become available as each one lands in the pipeline. The embedding-model head and
SetFit are available.

## A logistic head on a frozen sentence-embedding model

`train.method: embedding-head` (`byosm/train/embedding.py`) trains no encoder weight. It mean-pools the backbone's
last hidden states, the pooling sentence-transformers checkpoints such as all-MiniLM-L6-v2 (the default text backbone)
were trained with, and fits a multinomial logistic regression with `LOGISTIC_C` 1, the setting SetFit and the MTEB
classification protocol use, and the trainer's square-root inverse-frequency class weights. It is the floor a text
specialist can score in seconds before any fine-tune: frozen sentence encoders with a small head beat fine-tuned BERT
at 10 rows per label in [Casanueva et al.](https://arxiv.org/abs/2003.04807). It has not been measured in BYOSM yet.

The regression is written into the ordinary head, so the result is a plain `SpecialistModel` that the evals code
scores and `byosm export` turns into the usual ONNX graph (encoder, mean pooling, linear head), checked by `verify`.
The head's `LayerNorm` gets scale `1/sqrt(dim)` and no shift, so its output is the centred, unit-length embedding the
regression was fitted on (`fit_head`). A binary regression becomes the two logits `-z/2, z/2`, whose softmax is its
sigmoid, and a label with no training row gets the logit `ABSENT_LOGIT` (`logistic_weights`).

Pooled embeddings are cached by content in `embeddings.sqlite` under the byosm cache (`byosm/train/vectors.py`), keyed
by backbone, `max_len`, `truncation` and the text's SHA-1, so seed replicas and refine rounds encode a row once per
machine. Dev rows are scored through the model, as in every other training, so the selection path is the shipped one.

Gotchas: there is one pass, so `history` holds one entry whose loss is the regression's training log loss;
`epochs`, `pretrain_steps`, `ema`, `freeze_backbone`, `init_from` and `distill_from` have no effect. A student distilled
from an embedding-head teacher is fine-tuned (`byosm/distill.py`). The spec refuses the method for other modalities,
frame tasks and attention pooling. Fine-tuning and the embedding methods share how rows are split and seeded
(`prepare` in `byosm/train/rows.py`) and how a run is saved (`finish` in `byosm/train/outputs.py`), and the teacher
and embedding caches open their database with `cache_db` (`byosm/config.py`).
The Machine page's training speed per backbone counts fine-tuning runs only (`_fine_tuned` in `byosm/machine.py`):
one pass over cached embeddings would make a backbone look many times faster to fine-tune than it is.

## SetFit: label pairs before the logistic head

`train.method: setfit` (`byosm/train/contrastive.py`) first fine-tunes the encoder on sentence pairs built from the
labelled rows, then fits the same logistic head as `embedding-head` on the tuned embeddings. Each row gets
`pairs_per_row` pairs (40), half with another row of its label (target cosine 1) and half with a row of another label
(target 0), as in the paper's 20 iterations of one positive and one negative pair; the loss is the mean squared error
between the cosine of the two mean-pooled embeddings and the target. One pass over the pairs at `lr_pairs` (2e-5, the
paper's rate) with `WARMUP_SHARE` 10% warmup and cosine decay, at `batch_size` pairs per step, in fp32. SetFit beat
standard RoBERTa-large fine-tuning by 19.3 points at 8 rows per label and 5.6 at 64
([SetFit](https://arxiv.org/abs/2209.11055)), so it pays where a specialist has few real rows per label. It has not
been measured in BYOSM yet.

Gotchas: a big training set would make the pass slow for little gain, so the pairs are capped at `MAX_PAIRS`
(20,000) by giving each row fewer; a label with one row gets only different-label pairs. Tuned embeddings change with
every run, so only `embedding-head` reads the embedding cache. Seeds change which pairs are drawn, so seed replicas
differ, unlike `embedding-head`, whose replicas are identical.

## Embedding methods in the proposer's catalogue

Both embedding methods edit `train.method`, and a family used to be matched on the path alone, so a SetFit proposal
would have been tagged as an embedding head. A catalogue path can now pin a value: `train.method=setfit` claims only the
edits that set that value (`claims` in `byosm/improve/paths.py`), while `allowed` still checks the path without it, and
the proposer's prompt explains the `=v` form. SetFit also owns `train.pairs_per_row` and `train.lr_pairs`. A family can
name the task types it fits, so taggers are not offered either method (`task_types` in
`byosm/improve/techniques.py`).

A head on a frozen MLM encoder is the weak variant: on Banking77 at 10 rows per label, frozen BERT with a small head
reached 67.55 against 85.19 for frozen sentence encoders ([Casanueva et al.](https://arxiv.org/abs/2003.04807)). So an
edit that switches a specialist to an embedding method without naming a backbone also switches its encoder to
`EMBEDDING_MODEL` (all-MiniLM-L6-v2), or to `MULTILINGUAL_EMBEDDING_MODEL` (paraphrase-multilingual-MiniLM-L12-v2)
when the spec lists a language other than English, unless it already runs one of the `SENTENCE_ENCODERS`
(`byosm/spec/backbones.py`, `_embedding_encoder` in `byosm/improve/paths.py`). Both are mean-pooled
sentence-transformers checkpoints, the pooling the head uses. The switch happens in `apply_ops`, which `byosm exp
create --set`, proposals, ablations and Keep training all go through, so the stored spec always names the encoder that
ran. A backbone named in the same edit is kept, and later edits to an embedding experiment leave its encoder alone.

## Tool-choice pairs are written in several layouts, and calib and test can hold tools training never sees

The tool-choice student scored well on its own layout and far worse on test sets written in other layouts, and its
calib and test offered the same tools it trained on, so selection could reward memorising tool names. A specialist
made by `byosm options new` now carries a pair plan, `data.pairs` in `spec.yaml` (`PairPlan` in
`byosm/spec/sections.py`); a spec without one (every specialist made before) keeps reading its table as a plain local
table, with the same data key.

The table in `.byosm/options/<name>.csv` keeps, besides the rendered `text`, each pair's `request`, its tool as JSON
(`option`), its request `group` and its `layout`, so the data stage (`byosm/options/training.py`) can write the
training pairs again for whatever plan an experiment has, while dev, calib and test come from the table's `text`
unchanged, so every experiment is scored on the same rows. With the default plan the table is the same, row for row,
as the one written before pair plans existed.

- **Layouts** (`byosm/options/layouts.py`): `marked` (the request, then `Option: name: description (params)` and the
  other names, as before), `json` (the request, then the tool as a JSON schema), `openai` (an OpenAI-style request
  payload with the tool in `tools`) and `prose` (the tool described in words, then the request). `data.pairs.layouts`
  lists the ones training uses; each pair draws one from a random generator seeded by its request, tool and kind, so a
  rebuild writes the same rows.
- **Masked names**: `data.pairs.mask_names` is the share of training pairs whose tool names, the scored one and the
  others listed with it, become `tool_1`, `tool_2`, ..., so the model has to read the description and parameters.
- **Held-out tools** (`byosm/options/heldout.py`, `byosm options new --held-out`): a hash of each tool name puts a
  quarter of the names on the held-out side. A request whose called tools (or, when none fits, offered tools) include
  a held-out one goes to calib or test and keeps only its held-out tools; every other request goes to train or dev and
  drops any held-out tool from its offer. No tool name in calib or test appears in train or dev. Because held-out
  requests keep fewer of their offered tools, calib and test hold a larger share of fitting pairs than train.
- **Held-out layouts**: `data.pairs.held_out_layouts` (`prose` with `--held-out`) are drawn only for calib and test
  rows, which draw from the training layouts and these; the plan refuses a layout that is in both lists.

When the table marks calib itself, `carve_calib` keeps it and leaves dev whole; otherwise it still halves dev, now
keeping every pair of one request (`meta.group`) on one side. The `options rival` LLM now reads the same rendered
pairs as the model, one item per pair, and answers with item numbers, so it is compared on the same input in every
layout instead of a clean listing only it saw.

## Paraphrased requests are written once per machine

`data.pairs.paraphrases` (0 to 5) adds that many rewrites of every training request, written by the `generate` role
in batches of 20 (`byosm/options/rewrite.py`): other wording, order and tone, the same meaning and every value. Each
rewrite is paired with the request's offered tools and keeps their labels; its rows carry `llm:<model>` as source and
`paraphrase` as kind, and never reach dev, calib or test. Answers are kept by content in `llm-memo.sqlite` under the
cache folder (`byosm/llm/memo.py`), keyed by the prompt version, the writer's model, the instruction and the request,
and saved batch by batch, so a rebuild, a seed replica or another experiment with the same plan asks the LLM only for
requests it has not rewritten, and an interrupted run keeps what it got. A failed batch is logged and not kept.

## Near-miss negatives are always relabelled by the judge

xlam requests come with tools that are easy to rule out, and none of them comes without a fitting tool, so a pair
model rarely sees a negative that looks like a positive. `data.pairs.near_misses` lists the kinds that
`byosm/options/nearmiss.py` mines from every training request with a fitting tool, each one pair per request:

| Kind | The pair |
|---|---|
| `nearest` | the training tool most like the request and its fitting tool (TF-IDF over the tool text), added to the offer |
| `same_category` | a tool offered beside the fitting tool in another training request, as xlam has no category field |
| `missing_parameter` | the fitting tool, with the request rewritten by the `generate` role to leave out a required parameter |
| `removed` | the remaining tool most like the request, once every fitting tool is taken out of the offer |

Candidates come only from training requests, so a held-out tool never enters training this way. A near miss can fit
after all, so none enters training on the miner's guess: `relabel` (`byosm/options/relabel.py`) gives each one to the
`judge` role, written in the `marked` layout, takes the judge's label, and drops what it drops or fails to answer.
Configure `judge` as the strongest model available; its verdicts are kept by content in the same cache as the
paraphrases, keyed by its model, the task's labels and the pair, so a rebuild asks it only about new pairs. The kept
rows carry `near-miss:<judge model>` as source and their kind in `meta.kind`, and are written in the plan's layouts
and masking like every training pair.

## R-Drop, mixup and re-initialised top layers are options

Three second-tier regularisers are now `train` settings the proposer can choose (families `consistency` and
`reinit-top`), all off by default and applied to classification batches with gold labels only
(`byosm/train/regularise.py`):

| Setting | What it does |
|---|---|
| `train.rdrop` | runs each batch twice through the dropout-on model and adds this weight times the symmetric KL between the two predictions |
| `train.mixup` | Beta(alpha, alpha) mixes the pooled hidden states of each row with another row of the batch and adds the label-mixed loss |
| `train.reinit_layers` | re-initialises the top N encoder layers before training (normal with the model's `initializer_range`, LayerNorm reset), since recent transformers skip modules already flagged as initialised |

Distillation batches skip R-Drop and mixup. The evidence is from 2019 to 2021 BERT-sized models and the gains are
small, so both families carry low priors. Not measured on a specialist yet.

## Few real rows per label suggest SetFit

SetFit's largest gains are below about 30 labelled rows per label, so a diagnosis rule (`few_shot` in
`byosm/improve/rules_fit.py`) proposes `train.method: setfit` and `embedding-head` for a fine-tuned text classifier
with fewer training rows per label than that; as a rule, it raises those families in the bandit rather than deciding.
The proposal cost now scales with the method: the embedding head costs about 5% and SetFit about 30% of fine-tuning on
the same rows (`METHOD_COST` in `byosm/improve/cost.py`), rough ratios to be replaced by measured GPU hours.

## A small TPE sampler suggests numeric settings

Once a `train.*` setting of an eligible family has four or more distinct values tried on a specialist (experiments
that edited it, plus the promoted model), `byosm/improve/tpe.py` proposes the next value the way TPE does: the best
quarter of tries by calib forms one Gaussian density and the rest another, on a log scale when every value is
positive, with a bandwidth of the value span divided by the number of tries; of 24 draws around the good tries, the one
with the highest good-to-bad density ratio is proposed (`tpe-<path>`, source `tpe`), rounded to a whole number for
integer settings. It is written by hand because Optuna is not a dependency. Data-size settings are left out, since
they change the data rather than the training. The bandit decides whether a TPE proposal runs, like any other.

## The Techniques page shows the catalogue and what each family did

The research behind the catalogue only helps if it can be seen. The Techniques page (sidebar, under Build) lists every
family of `byosm/improve/techniques.yaml` with its status (available, built-in or planned), the kinds it fits, its
evidence links, and, across every specialist, how many experiments it ran, how many ended in a sure lead and their
GPU time (`catalogue_board` in `byosm/improve/bandit.py`, `GET /api/techniques`). A bar chart compares tries with sure
leads for the families that have run, and tabs filter by status. The two systems endpoints moved from
`byosm/app/server.py` into `byosm/app/systems_routes.py` with the rest of the systems routes.

## Forecaster pretraining v2 is a recipe

The patch transformer is trained by a named, frozen recipe (`Recipe` in `byosm/series/recipe.py`), so a new way of
pretraining is a new entry in `RECIPES`, not a change to the old one. `v1` is the recipe every base so far was built
with, and it is reproducible bit for bit: same windows, targets and random draws (a test rebuilds its batches the old
way), same weights files (`patch-base.pt`, `patch-large.pt`), same model name and the same resume key. `v2` gathers
what the published ablations rank highest (TiRex, Chronos-2, Moirai 2.0; see the experiment search report):

| Field | v2 | Why |
|---|---|---|
| `quantiles` | 21 levels, 0.01, 0.05 to 0.95 in steps of 0.05, 0.99 | Moirai 2.0's largest single gain; scoring and served forecasts stay on the 9 standard levels, taken as a subset (`served`) |
| `mask_rate` | `MASK_RATE` 0.15 | One contiguous span of whole input patches is hidden per window, its length uniform between 0 and twice the rate, so 15% of patches on average; a short window the span would hide entirely is left whole; TiRex reports -7.6% CRPS |
| `amplitude`, `censor`, `mixup` | chances 0.3, 0.1, 0.3 | Amplitude modulation (the deviation from the context mean times a piecewise-linear gain between 0.5 and 1.5), censoring (clipped from below or above at a quantile of the context up to 0.3), and TSMixup of two series of the batch, each scaled by its mean absolute context value, with a Beta(1.5, 1.5) weight; TiRex reports a further -4.4% |
| `arcsinh` | on | After the window's z-score, inputs and targets go through arcsinh, and forecasts back through sinh, which is monotone, so quantiles stay ordered (they are still sorted) |
| `synthetic` | `SYNTHETIC_SHARE` 0.2 | A share of every pretraining batch comes from `byosm/series/synthetic.py`, so pretraining runs with no downloaded corpus |
| `sizes` | base, then long | The base size first, so v2 is compared with v1 at the same size; the 2048-point `long` size is v2's next step |

The augmentations act on the whole window, context and target together, before scaling, and masking acts on the
context only, before the window's mean and spread are taken, as missing values would. They also run when a v2 base is
fine-tuned on a domain; synthetic series are pretraining only. A v2 base is saved as `patch-<size>-v2.pt` and its
fine-tuned model is named `patch transformer, pretrained v2`, so both recipes can sit side by side on the board.

The synthetic generator draws a sampling frequency (5 and 15 minutes, hourly, daily, weekly, monthly, quarterly) and
its natural seasonal periods, then adds up to three harmonics per period, a linear and quadratic trend, AR(1) noise,
a random walk (30% of series), Poisson level shifts and a random level and scale, and floors 20% of series at zero.
Lengths are uniform from 128 points to the context plus the horizon, so short histories are seen too. It is seeded by
the training generator, so a pretraining run is reproducible.

Cost, measured on the CPU at batch 256: one v1 base step takes 524 ms forward and backward and 8 ms to build its batch;
a v2 base step 483 ms and 19 ms (10.4M parameters, most of the extra in the 21-level head); a v2 long step 1,012 ms
and 25 ms (18.2M). `series pretrain --recipe v2` pretrains it; nothing has been measured on GIFT-Eval yet.

## Quantile boosting reads the calendar

GIFT-Eval series keep their start time (`Series.start`); every window carries it, `Prepared` keeps a `Clock` (start
and frequency, `byosm/series/calendar.py`) per training history, and every forecaster's `predict_many` takes one clock
per history, or none. Wrappers pass them on. `MultiRate` passes them at the native rate only: a coarse rate already
reads coarse lags as native ones, so it forecasts without dates. Live forecasts of an example get the date of its first
value; a pasted series has none.

`calendar.features` gives the minute, hour, day of week, day of month and month of every step. GIFT-Eval's frequency
aliases (`H`, `10S`, `15T`, `W-SUN`, `A-DEC`) are stepped with numpy, since pandas 3 no longer reads most of them;
monthly and longer steps land on the first of the month. An unknown date is -1, not NaN: scikit-learn's histogram
binning fails on a column with no value at all, which a dataset without dates would give.

`QuantileBoosted` (`gradient boosting, quantile`, in `byosm/series/models.py`) learns from the same rows and target as
boosting (lags scaled by the recent level, minus the seasonal base, divided by the noise), with the calendar of each
forecast step added as known-future covariates, and fits one `HistGradientBoostingRegressor(loss="quantile",
quantile=q)` per level instead of one squared-error model plus residual quantiles taken on the last 10% of origins.
That is nine fits instead of two. It is cached by content like boosting. The boosted rows moved to
`byosm/series/rows.py` and the squared model builds the same table and forecasts bit for bit, so its cache stays
valid; the classes stay in `models.py`, because saved versions pickle them by module path.

A sanity check, not a benchmark: on 4 hourly series with a weekend drop whose starts are 13 hours apart, 30
iterations, the median's error is 0.595 with dates, 0.865 for the same model without them and 0.842 for squared-error
boosting. fev-bench's tuned LightGBM beats Chronos-2 on 17 of 100 tasks with quantile objectives and covariates; it has
not been run on the GIFT-Eval forecasters yet.

## Keep training escalates the recipe, not only the steps

A forecaster version records two sets of options: `predict` (multi-rate, routing), which only change how saved models
forecast, and `fit` (`FitCfg` in `byosm/series/prediction.py`: `pretraining`, the recipe of the pretrained base, `v1`
by default, and `quantile_boosting`, off), which change which models are trained. The catalogue maps
`fit.quantile_boosting` to `quantile-boosting` and `fit.pretraining` to `forecaster-pretraining` (turned on as `v2`),
next to `predict.rates` and `predict.routing`, and `next_technique` picks among all four, best prior per cost first.
The board's Options column lists both sets.

A technique round now refits only what its option changes and reuses every other saved model. `candidates` builds the
models a training round would fit under some fit options (`QuantileBoosted` instead of `Boosted`, the pretrained
model of the options' recipe), and `refreshed` (`byosm/series/fit.py`) compares the candidates under the promoted
version's options with those under the new ones: models whose name disappears are dropped from the saved ones, models
whose name appears are trained at the promoted effort, and the rest are only scored again. A prediction-time
technique changes no name, so its round trains nothing, as before. When a technique turns on a recipe whose base has
no weights yet, the round pretrains it first (`base_for` in `byosm/series/escalation.py`).

A technique that costs more than a doubling of the training (`PLATEAU_COST`, 2 in the catalogue's units) is tried only
once a plateau is stored, that is when the existing escalation would pretrain a missing base or double the steps.
Forecaster pretraining v2 costs 5, so after a plateau the next Keep training pretrains the v2 base (when missing),
fine-tunes it on the domain at the promoted effort and scores it beside the reused models, once per plateau: the
version records `fit.pretraining: v2`, so the round is not repeated at that effort. Only then does the old escalation
run. Quantile boosting costs 0.5 and is tried right after multi-rate and routing, before any doubling.

The plateau's upstream (the largest pretrained base and the corpus size) and the next lever are now read for the
promoted version's recipe (`recipe_of`). If v2 wins on calib, its base differs from the one stored with the plateau,
so the next round doubles the steps on top of v2 instead of pretraining a bigger v1 base nobody would use; if it
loses, the upstream is unchanged and the old escalation pretrains the next v1 size. A recipe has its own size ladder:
v1 goes base then large, v2 base then long, and the lever names `--recipe`. None of this has been run on the GIFT-Eval
forecasters yet.

## More of the same data waits for the learning curve

"Add more data" had run seven times on plateaued specialists without anyone checking whether data was the limit.
`byosm curve <name>` (`byosm/curve.py`, also the "Measure the learning curve" next step when the data check says thin)
trains the promoted recipe on nested subsets of 25% and 50% of its training rows (`train.data_fraction`, a prefix of
the rows sorted by a hash of their id, so the smaller subset sits inside the larger), fits calib against the log of the
row count by least squares over the two subsets and the full run, and stores the slope times ln 2 as the gain expected
from twice the data. The `more-data` rule stays silent while the stored curve of the promoted model says that gain is
below the seed noise (or 0.002 when the noise is unknown). Curve runs are auxiliary: they never promote and are hidden
from the Runs tab. New sources found by the dataset search are still offered, since they bring other data rather than
more of the same.

## A student learns from the average of its teacher's seeds

Distilling an ensemble of finalists into one student is worth about as much as the ensemble without its serving cost.
A distilled student now takes its soft targets from the teacher and up to `train.distill_seeds - 1` (2 by default)
of the teacher recipe's trained seed runs, averaged row by row (`teachers` and `averaged` in `byosm/train/teacher.py`);
a teacher with no seed runs teaches alone as before, and `distill_seeds: 1` restores the single teacher. Since a
recipe has seed runs only after it reached the finalist vote, this is the merge step for the seeded finalists. Not
measured on a specialist yet.

## Self-training on the unlabelled pool

With `train.self_train` above 0 (a `self-training` family for text classifiers, off by default), training first lets
the promoted model label the specialist's unlabelled in-domain pool (the same pool distillation reads, capped by
`train.distill_unlabeled`, never rows already in training) and adds the rows it labels with at least that confidence
as ordinary training rows with source `pseudo` (`pseudo_labels` in `byosm/train/teacher.py`). The gains reported are
largest when a model starts with few labels. With this, every family in the catalogue is available or built in: the
small hyperparameter search is the TPE proposals together with short-run screening. Not measured on a specialist yet.

## A floor baseline that answers the positive label for everything

`always-positive` (`byosm/predictors/constant.py`) gives every input the positive label with certainty and has no
parameters. Its score follows from the labels alone: on a pair suite it chooses every offered tool, on a turn suite it
ends the turn at the first moment allowed, which is a plain silence timeout. That makes it the shared trivial baseline
of the protocol check that every public suite passes before it is added: our conversion and scoring must give it the
value the labels imply, or the benchmark's own published number for the same baseline. It is not added to a spec's
baselines, so it never becomes a specialist's "best outside baseline".

## Pair suites are scored per request, and the board says how many rows each model was scored on

A pair suite scores one (request, tool) row at a time, but what a user of the decision sees is the set of tools chosen
for a request. Whenever a suite's rows carry the request they belong to (`meta.group`) and the specialist's labels are
`fits` and `no_fit`, `score` (`byosm/evals/scoring.py`) also groups the calibrated decisions by request
(`byosm/evals/grouped.py`, reusing `evaluate` from `byosm/options/select.py`): `exact` (exactly the fitting tools
chosen), `tool_needed` (the same, on requests where a tool fits) and `no_tool` (nothing chosen, on requests where none
fits), each with a bootstrap interval, for the whole suite and per `meta.category`. On BFCL, `no_tool` over the
irrelevance requests is BFCL's own irrelevance score. Builtin tool-choice suites get the same numbers, since their rows
already carry the request.

`byosm board` shows the test rows each model was scored on beside its score on every suite (`<suite>:n` in
`board_rows`), so a rival scored on a smaller or older version of a suite stands out, and `--detail` adds the
per-request summary for suites that have one.

## False cutoffs at 300 and 600 ms, scored the way eot-bench scores them

Recall at 98% precision on single transcripts hides the trade-off a voice agent lives with: answering sooner after the
user stops cuts more users off mid-sentence. LiveKit's eot-bench ([github.com/livekit/eot-bench](https://github.com/livekit/eot-bench),
`eot_harness/metrics.py`) reports instead the lowest false-cutoff rate a detector reaches within a mean-latency
budget, and `byosm/evals/cutoffs.py` ports that sweep for suites whose rows carry silence timing:

- A suite row is one state of a silence span: `meta.span` names the span, `meta.span_s` is its length and `meta.at_s`
  the silence (seconds) after which this text is what the detector reads. The span's label is the turn's end
  (`eot`, the positive label) for the last span of a turn and a pause (`hold`) for the others.
- For each threshold (0 to 1 in steps of 0.01), a span fires at the first state whose chance is above it. A policy adds
  an action delay (0.2 to 1.0 s, the earliest the agent may act) and a timeout (1.0 to 3.5 s, when it ends the turn
  anyway). A pause is cut off when the agent acts or times out before the pause ends; an end's latency is when the
  agent acts, or the timeout when it never fires. These grids are eot-bench's defaults.
- `cutoffs.all` is eot-bench's own protocol: over every row of the set (calib and test, the split eot-bench publishes
  on), the policy with the lowest cutoff rate among those whose mean latency is within 300 ms, and within 600 ms, with
  eot-bench's tie-breaking. These numbers are comparable with eot-bench's published tables. The policy is tuned on the
  rows it reports, as eot-bench does.
- `cutoffs.held_out` picks the policy on calib only and reports its cutoff rate and latency on test, the number
  selection-honest comparisons should use.

`byosm board --detail` shows the held-out cutoff rates. A detector that always answers "finished" fires at once in
every span, so its policy reduces to a silence timeout: its cutoff rate at a budget is the share of pauses longer than
the budget, which is eot-bench's published "VAD baseline" and the protocol check of the eot-bench suite.

## A suite names the public dataset it came from and its licence

A suite made from a public benchmark is frozen as a `local` suite, so its rows no longer point back at the Hub the way
an `hf` suite's `source` does. `EvalSuite.dataset` and `EvalSuite.licence` record the Hugging Face dataset and the
licence on its card when the suite is written, and `EvalSuite.origin` (the dataset, or an `hf` suite's source) is the
one place the model card reads: `suite_dataset` names it in the results tables, `load_facts` lists it among the
card's `datasets`, and the caveats list each public suite with its licence. `byosm suite ls` shows it too. The card
licence lookup moved from the data release to `licence` in `byosm/data/hf_source/cards.py`, next to the other
dataset-card reads, so the release card and new suites read licences the same way.

## BFCL and When2Call become frozen tool-choice suites

The tool-choice specialist was only scored on pairs written from xlam-60k, where an LLM rival is near its ceiling.
`byosm suite public <name> [benchmark ...]` (`byosm/cli/benchmarks.py`, `byosm/benchmarks/`) adds public sets where
strong models are not: without names it adds every catalogued benchmark that fits the specialist and is missing. Each
is converted by one module per source into rows in `.byosm/benchmarks/<name>/<suite>.jsonl`, added as a `local` suite
with its `dataset` and the `licence` on its card, frozen on the first `byosm eval`, and from then on scored for every
experiment and rival by the same code as the builtin suite. A pair benchmark fits a specialist whose labels are `fits`
and `no_fit` with `fits` positive.

| Suite | Source on the Hub | Licence | Requests kept | A pair fits when |
|---|---|---|---|---|
| `bfcl` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, v3 `simple`, `multiple`, `irrelevance` | Apache-2.0 | every request that offers a function and whose answer calls only offered ones | the possible answer calls that function; never on `irrelevance` |
| `bfcl-live` | the same, `live_simple`, `live_multiple`, `live_irrelevance` | Apache-2.0 | the same | the same; never on `live_irrelevance` |
| `when2call` | `nvidia/When2Call`, `test/when2call_test_mcq.jsonl` | CC-BY-4.0 | every item that offers a tool | the right answer is `tool_call` and it is the target tool |

- **Request text.** BFCL questions are chats: the request is the user messages of the first turn; system prompts are
  dropped. When2Call's `question` is used as is.
- **Functions** are JSON schemas; `option` (`byosm/benchmarks/pairs.py`) turns one into the option the layouts read:
  its properties become parameters, each marked `required` from the schema's list.
- **Layouts.** `pair_rows` writes one row per offered function with `pairs_of`, rendered unmasked in the layouts the
  specialist's own calib and test use (`split_layouts(plan, "test")`), so the suite tests the layouts it was built
  for and its held-out ones.
- **Split.** A hash of the request id puts a fifth of the requests in calib (`split_of` in
  `byosm/benchmarks/source.py`, on `hashed` from `byosm/options/heldout.py`, shared with the held-out tool split). When2Call hashes its BFCL source question instead, so the three variants of one
  question (call, ask, cannot answer) land on the same side; since BFCL ids are hashed the same way, a question that is
  in both `bfcl-live` and `when2call` sits on the same side of both.
- **When2Call as binary** asks "should the model call this tool now": `cannot_answer` items (the tool was removed) and
  `request_for_info` items (a required value was removed from the question) are `no_fit` for every offered tool, as
  the `missing_parameter` near miss guesses. The per-category request scores keep the three kinds apart. Items that
  offer no tool give no pair.
- **Left out.** BFCL `live_relevance` names no fitting function, so its pairs cannot be labelled. BFCL v4 is only in
  the GitHub repository; the Hub has v3. ToolRet (`mangopy/ToolRet-Queries`) lists only each query's target tools, with
  no offered set and no negatives, is scored as retrieval over a 43k-tool corpus, and its card states no licence, so
  turning it into pairs would mean inventing the negatives.

**Protocol check** (`run` in `byosm/benchmarks/protocol.py`, with the pair expectations in `expected` in
`byosm/benchmarks/pairs.py`). Before a suite is added, `always-positive` is scored on the frozen
rows by `score` and must reproduce what the labels imply on test: precision equal to the share of fitting pairs, recall
1, F1 `2b/(1+b)`, `tool_needed` equal to the share of tool-needing requests whose every offered tool fits (BFCL
`simple`), and `no_tool` 0, which is BFCL's own definition: a model that always calls a function scores 0 on
irrelevance. Shares are compared within 5e-5, as request accuracies are rounded to four places. The outcome is kept as
the `protocol:<suite>` report with the rows it was scored on; a mismatch, or a suite with no test rows, discards the
frozen rows and leaves the spec unchanged.

## eot-bench becomes a turn-completion suite, checked against its published silence timeout

`byosm suite public <name> eot-bench` fits a binary text specialist whose positive label is a "turn over" name
(`FINISHED_NAMES` in `byosm/predictors/turn.py`) and whose first language is one of eot-bench's 14
(`byosm/benchmarks/eot.py`).

- **Source.** `livekit/eot-bench-data` (CC-BY-4.0 on its card; the harness is Apache-2.0), the validation file of the
  spec's language. Only the `id`, `silence_spans` and `words` columns are read, through `HfFileSystem` and a pyarrow
  column projection, so the audio in the same file is never downloaded.
- **Rows** follow eot-bench's harness (`eot_harness/io.py`): a span shorter than 0.1 s is skipped; the last span of a
  turn is its end and every other one a pause, kept only between 0.2 and 5.0 s as the metric does; each span is read
  every 0.1 s from its start, and at time t the detector hears the words that ended at least 0.5 s earlier
  (eot-bench's transcript lag). Each distinct text in a span is one row, at the first moment it is heard, which scores
  a text-only model exactly as reading it at every step would. The text is the current turn's words only; the earlier
  conversation in `messages` is left out, as turn-completion reads the turn so far. A moment with no word heard yet
  is an empty input, scored like any other.
- **Split.** A hash of the turn id puts a fifth of the turns in calib. The suite keeps the row-level objective (recall
  at 98% precision) and adds `cutoffs`; a row's label is its span's, so an end read before its last words arrive is
  still an end.
- **Protocol check.** `always-positive` reduces to a silence timeout, so on every row of the file (the split eot-bench
  publishes on) its cutoff rates at 300 and 600 ms must equal eot-bench's published "VAD baseline" for the language,
  read from `output/.../language_comparison/summary.json` at a pinned commit (`COMMIT`) and fetched once into the
  open-data cache. A language with no published baseline is refused. Once the check passes, every model's published
  cutoff rates on that file are kept in the `protocol:eot-bench` report and printed, as references measured on the
  same rows.
- **Not added.** The Smart Turn v3.2 test set (`pipecat-ai/smart-turn-data-v3.2-test`, CC-BY-4.0) is audio only: its
  `spoken_text` column is empty and it has no word timing, so a text specialist cannot read it without a speech
  recogniser.

## A public suite's dataset is never searched as training data

The training-data search (`find_datasets` in `byosm/dataset_search.py`) already skipped the datasets behind `hf`
suites. It now skips every suite's `origin`, so the BFCL, When2Call and eot-bench files a `local` public suite was
written from are never offered to the picker as training rows for the same specialist.

## The hard public benchmarks are a next step

Adding BFCL, When2Call or eot-bench took a command nobody would know to type. The next steps now offer "Score on hard
public benchmarks" (`suite public <name>`) whenever a public benchmark fits the specialist's labels and language and is
not one of its suites yet (`_public` in `byosm/next_steps.py`), so the ready loop adds them, after their protocol
check, before it judges the rival and robustness checks.

## New test cases are a mandatory part of the overfitting check

A model can learn the quirks of its own benchmark (its layout, its source, its label conventions) and still score
well on that benchmark's test split. The overfitting check now has a sixth finding, "New test cases"
(`byosm/evals/fresh.py`): the drop from the primary suite's test score to the score on an LLM-written suite
(`byosm testset`, rows written for the task and deduplicated against the training data) is ok up to 0.05, a warning up
to 0.15 and a failure beyond. For classifiers it is mandatory: while the promoted model has not been scored on such a
suite, the "No overfitting" health check stays unknown, which blocks production readiness, and its step writes the test
set ("Write new test cases"); once the suite is scored, the step runs the check again. Kinds that cannot write test sets
yet (taggers, forecasters) are judged without it, and the report lists the check as skipped. The drop also reflects
how much the written cases differ from the benchmark, so a warning is a prompt to read the failures, not proof of
memorisation by itself.

## The lab lets an LLM write training methods that a person approves

The catalogue can only combine techniques someone already built. The lab (`byosm/lab/`) lets an LLM write a new one
as code, in the spirit of a coding agent, with a person in the loop:

| Step | What happens |
|---|---|
| Write | `byosm lab write <name> [--ask ...]` or "Write a method" on the specialist's Lab tab: the `lab` LLM role reads the task, the promoted model's scores, weakest labels and confusions, the experiments tried and every earlier method with its outcome or error, and returns a name, a hypothesis and a module (`author.py`) |
| Contract | the module defines any of `rows(train, rng)`, `loss(logits, targets, weight, step)` and `after_step(model, step, total)` (`contract.py`); the architecture, tokenizer, scoring and export stay the pipeline's, so a method's model is scored and shipped like any other |
| Checks | an AST check (`static.py`) refuses imports outside torch, numpy, math, random and a few standard modules, calls such as open, eval and exec, dunder names and attributes, torch.hub, torch.load and torch.save, and any statement other than imports, constants and functions at import time |
| Approve | the draft id is the hash of its code; nothing runs until a person approves that exact code (`decide`, or Approve on the Lab tab); code that fails its checks cannot be approved, and the loader refuses an unapproved, changed or failing draft (`plugin.py`) |
| Run | `byosm lab run` trains the promoted recipe with `train.lab_method` set (`run.py`), as an experiment of the `lab-method` family (status `manual`, so the proposer can never set it), through the same seed-averaged vote; its error or result goes back on the draft, and the next write reads it |

Loaded code gets a reduced set of builtins and an import guard, but this is not a sandbox: the approval is the real
gate, and container isolation of jobs is the next layer. A method that wins is still an experiment: to become a
default it has to be rewritten into the pipeline with tests, like any other result.

## Added public benchmarks are scored as a next step too

`suite public` only adds a suite; the scores come from `byosm eval`, and the robustness check asks for that only while
the model has no other suite scored. Once every fitting public benchmark is a suite, `_public` in
`byosm/next_steps.py` now offers "Score on hard public benchmarks" as `eval <name> --suite <each>` for the public
suites the promoted model has no result on, so the ready loop (and the measurement plan) adds a benchmark and then
scores it, and a newly promoted model is scored on them as well.

## Measure everything: one plan of the missing measurements, run as jobs

Nothing of the research plan above had run on a real specialist, because each piece had its own command.
`byosm measure [--all | <name>...] [--dry-run]`, `GET /api/measure` and `POST /api/measure` (the "Measure everything"
button on the Jobs page) list and queue every measurement still missing for the promoted models
(`byosm/measure/plan.py`). Nothing is decided twice: a model's measurements are its next steps
(`suggestions` in `byosm/next_steps.py`) whose id is one of these, and a forecaster's comes from its saved versions.

| Measurement | Comes from | Command | Lane |
|---|---|---|---|
| Overfitting | the `overfit` health check: no report for the promoted model, or one for another model | `overfit <name>` | CPU |
| Public benchmarks | `_public`: a fitting public benchmark is not a suite yet, then a public suite has no result for the promoted model | `suite public <name>`, then `eval <name> --suite ...` | CPU |
| Seed noise | the `seeds` health check: fewer clean seed runs than `finalist_seeds`, or no spread yet | `seeds <name>` | GPU |
| Learning curve | `_curve`: the data check says thin and the stored curve is for another model | `curve <name>` | GPU |
| Pretraining v2 | no version of the forecaster has `fit.pretraining: v2` (`tried` in `byosm/series/escalation.py`) | `series pretrain --recipe v2` while no v2 base exists, then `series run <domain> --until-rival --rounds 2` | GPU |

Archived specialists are left out, and so is a specialist whose next steps fail to compute (as on Home).

**Order.** CPU cells first, by measurement then specialist; then GPU cells grouped by specialist. Every CPU cell is
submitted at once and the runner's own admission bounds them. A GPU cell starts only when no GPU cell of another
specialist has a job (`gpu_turn` in `byosm/measure/conduct.py`), so two specialists never share the GPU queue and one
specialist's trainings finish before the next starts; the cells of one specialist (seeds and curve) may run together.

**The run** is one workspace report (`measure`): the cells in order, each with its current job, every job it ran and
its state (open, done, failed). `byosm measure` and the POST only merge the fresh plan into it: an open cell is kept
as it is, a finished cell that is still missing is opened again, new ones are appended. The app's `Conductor` ticks
every `TICK_S` (5 s, and at once after a POST) and for each open cell: leaves it while its job is queued or running;
fails it when the job failed, was interrupted or cancelled; otherwise reads the cell's step again from the plan rules
for that specialist, so a measurement done meanwhile, or by its first job, ends the cell as done, and the next step
of a two-step measurement (scoring after adding the suites, running after pretraining) is the new command. A cell
whose measurement is still missing after `MAX_JOBS` (3) jobs fails instead of looping. A job already queued or
running with the same command is followed, not started again, and nothing is ever cancelled. Writes compare a
version number, so a CLI queue and a tick cannot overwrite each other. When no app serves the workspace, the queued
run waits for `byosm app`, which resumes it, as it resumes the jobs themselves.

**The board** (`board` in `byosm/measure/run.py`) shows one cell per specialist and measurement: a run cell while it
is open, or failed on the same promoted model; otherwise the fresh plan's cell as `todo`; otherwise the run cell's
final state. The UI takes the running and pending states live from the job events and reloads the board on a
`measure` event, which the conductor publishes whenever it changes the run.

Not measured yet: nothing in this note has run on the workspace's specialists. `series run --until-rival --rounds 2`
tries v2 only once more training stops helping (the plateau rule above), so a forecaster may need its second or
third job, or fail its cell as still missing, before v2 is scored.

**On the Jobs page**, "Measure everything" is the primary action (top right) and posts the plan; the Measurements
card below it is the board: one row per specialist or forecaster, one column per measurement that appears, each cell
to do, waiting, running (spinner), done or failed, with the reason (and the failure) in its tooltip and the job's log
a click away, and the share finished as a progress bar. The button is never disabled: clicking it with nothing
missing says so, and clicking it during a run only adds what is not queued yet. Running and waiting come live from
the job events; the board reloads on each `measure` event.

## A scout proposes techniques from recent papers

The catalogue only knew the techniques of one research report. The scout (`byosm/scout/`) keeps it current: every
`scout.every_days` (7 by default, in `byosm.yaml`) the app queues a `byosm scout` job, and "Find new techniques" on the
Techniques page runs one at any time. A scan reads the newest arXiv papers for four areas (small text classifiers,
tool selection, forecasting with pretrained models, turn detection; `AREAS` in `papers.py`) and Hugging Face's daily
papers, skips papers it has read before, and asks the `scout` LLM role for techniques a small-model pipeline could
adopt, with the gain each paper reports. A proposal is kept only when it cites one of the papers just read, by its exact
URL, so links cannot be invented, and only when it does not overlap a catalogue family. Proposals wait on the Techniques
page; an accepted one can be sent to a text classifier's lab, which asks the LLM to implement it from that paper as
code the owner approves before it runs. The claimed gains are the papers' own numbers, not BYOSM's: a technique counts
here only once a lab run of it wins the seed-averaged vote. The scheduler checks once an hour and never runs a scan
while another is queued or running. Its state is a workspace report (`byosm/db/workspace.py`).

## Each technique family's effect on calib is measured against the parent experiment

Tries and sure leads say how often a family won, not by how much. `byosm/improve/effects.py` measures every experiment
tagged with a family against the experiment it was built from (`parent` in its proposal), and `catalogue_board` adds
the result per family as `calib_effect` (`GET /api/techniques`):

| Step | What |
|---|---|
| Pair | the experiment and its parent on the primary suite, both with a calib score and the same suite fingerprint. Any other experiment (no parent, parent not scored, another suite version, no calib split) is left out and counted in `left_out` |
| Effect | calib score minus the parent's, with the paired bootstrap 95% interval of `evals.compare` (1,000 draws over the same calib rows) when both sides kept calib units |
| Cache | report `effect:<exp>`, keyed by the parent and both sides' fingerprint and identity, so the bootstrap runs once per scoring. The promotion vote (`_record` in `byosm/promotion.py`) fills it after every build |
| Family | `measured`, `mean` of the deltas over every specialist, and a Student t 95% interval of that mean: mean ± t(0.975, n - 1) x sd / sqrt(n). Under `MIN_MEASURED` (2) there is no interval |
| Kind | the same summary per modality and task type (`kinds`), and every measured pair (`experiments`) |

The interval is Student t, not a bootstrap across experiments, because a family has two to five measured experiments
for a long time, and with so few a percentile bootstrap shrinks to the range of the values and covers too little. Each
delta already carries its own seed and sampling noise, so the spread across experiments includes it. Deltas are
differences in each specialist's own objective (all on a 0 to 1 scale), so a family mean pools metrics; the `kinds`
rows keep like with like. Ablation arms count like any other experiment, so their "off" arm adds seed noise around
zero. `scipy` (already installed with scikit-learn) is now a declared dependency for the t quantile.

Measured on this machine: one paired bootstrap takes 0.06 s for 2 labels and 1,000 calib rows, and 11.6 s for 77
labels and 1,540 rows, hence the cache. With 109 older experiments that made the first page load run for more than
five minutes, so the Techniques page now reads only the deltas, which are a subtraction, plus any interval already
cached; the paired interval of each experiment is computed by the promotion vote, once, when the experiment is built.
The family intervals use only the deltas, so they are complete from the first load.

The Techniques page draws it as a forest plot, "Calib effect per family" (`ui/src/pages/EffectChart.tsx`): one row
per family with an interval, a dot at the mean on a line across the interval, a dashed line at zero, sorted by mean.
Hovering a row gives the mean, the interval, how many experiments were measured and left out, and the mean per kind
when there are several. Families that ran but have fewer than two measured experiments are listed under the plot as
"Not enough runs", and the table repeats the mean and interval in a Calib effect column. The marks use `--chart-2`:
`--chart-1` is a light grey that gives a 2 px line about 1.5:1 contrast on the light card. The marks do not animate
in. The row height and label width of the horizontal charts now come from `ui/src/chart-size.ts`.

## Every job runs isolated: a limited process, or its own container

`jobs:` in `byosm.yaml` (`job_settings` in `byosm/config.py`, limits in `byosm/sandbox/limits.py`) sets how
`start` in `byosm/sandbox/launch.py` launches a job. Limits are applied once, at start: changing them never touches a
running job, and nothing is cancelled to apply them.

**Process mode** (`byosm/sandbox/process.py`, the default). The job starts as before, in its own session and process
group, and the app then sets its `nice` level (10), so the app and other jobs stay responsive when one job takes every
core. On Linux it also caps the address space (`RLIMIT_AS`) at `memory_gb` (default 75% of RAM), except for CUDA jobs,
whose driver reserves far more virtual memory than it uses, and the CPU time when `cpu_hours` is set (off by default,
since long trainings must finish). macOS enforces neither rlimit, so a Mac job gets the nice level on top of the GPU
memory share it already had (`byosm/train/gpu.py`). The applied limits are the job's `isolation` field in the Jobs
API, kept in the `job_isolation` table. Not measured: the nice level should cost nothing on an idle machine, but no
training was timed before and after it.

**Container mode** (`isolation: container`). The runner still sees one detached process: `start` spawns
`byosm/sandbox/supervise.py` under the usual exit-code wrapper, with a plan holding the image, the network rules and
the `docker run` arguments. So the log, restarts and cancel work as for any job. The supervisor builds the image if it
is missing (its output goes into the job log), starts the job's network and proxy, runs `docker run` attached with
the job's output into the log, and always stops and removes the containers and the network at the end. Cancel sends
SIGTERM to the supervisor, which stops the container with a 5 s grace (`STOP_GRACE_S`).

| piece | what it is |
|---|---|
| image | `docker/train.Dockerfile` built by the same `build_image` as served models, from the frozen code snapshot plus `pyproject.toml`; tag `byosm-train:<snapshot>-<hash of flavour, pyproject, Dockerfile>`, so one build per code version, and the dependency layer sits below the code layer. CPU torch, or PyPI's CUDA wheels for CUDA jobs |
| container | `--init`, read-only root, `/tmp` tmpfs (exec allowed: Triton and Inductor load compiled kernels from it), no capabilities, `no-new-privileges`, `--cpus`, `--memory`, `--shm-size 2g` for DataLoader workers, `--pids-limit 16384`, the host user's uid, `--gpus` only for CUDA jobs, `--log-driver none` since the job log is the record |
| files | the workspace, the BYOSM cache, the Hugging Face home and any `HF_*_CACHE` folder outside them, mounted at their host paths, so paths in the database stay valid and GPU slot locks (`byosm/train/gpu.py`) are shared with every other job. A file outside these folders is invisible to the job |
| environment | the job's own variables, the keys, `HF_*`, `DATABASE_URL`, `BYOSM_DEVICE`, `BYOSM_CONCURRENCY` and `BYOSM_LLM` pass by name (`--env NAME`), so no secret is on a command line; the app's token is not passed |
| exit code | the wrapper inside writes the job's code to `<id>.inner`; a container killed for memory has none, so the supervisor reports 137 and says the job used all of its memory |
| watchdog | reads the CPU of the container's processes (`docker inspect` pid), not the idle supervisor's |

**The network is an allow-list** (`byosm/sandbox/egress.py`, `allow.py`, `containers.py`). Each job gets its own
`--internal` Docker network, which has no route out, and a proxy container on that network and the default bridge.
The job's `HTTP(S)_PROXY` point at it; it tunnels CONNECT and forwards plain HTTP only to `huggingface.co` and `hf.co`
with their subdomains, the host and port of every LLM role (its `base_url`, or the provider's API), and the hosts in
`jobs.network`. `localhost` in a rule means the machine (`host.docker.internal`), so a local Ollama keeps working.
Refused hosts are listed in the job log at the end. Gotcha: on Linux the local LLM server must listen on the Docker
bridge, not only on 127.0.0.1 (for Ollama, `OLLAMA_HOST=0.0.0.0`). Gotcha: open-data portals, GitHub (Silero, GIFT-Eval
results, model prices) and datasets hosted outside the Hub are not on the default list.

**A container job falls back to a process, and says why.** `blocker` in `byosm/sandbox/choose.py` returns the reason,
written as the first log line and kept in the job's `isolation.fallback`:

| reason | why it blocks |
|---|---|
| the job would train on the Apple GPU | Docker on macOS runs Linux in a VM that cannot reach MPS |
| Docker runs in a VM (macOS, Docker Desktop on Linux) | SQLite's WAL shares memory between processes on one kernel; across the VM boundary that is unsafe |
| no Docker CLI or daemon | nothing to run it in |
| a CUDA job and no NVIDIA runtime or hook | `--gpus` would fail |
| `DATABASE_URL` is a server | the internal network cannot reach it |
| an LLM role is `claude-code` or `codex` | they are command-line tools on the host |

Every job counts as a GPU job, because commands like `auto` decide at run time whether they train. Not verified: all
of this is tested with Docker mocked; no container job has run on a Linux machine with an NVIDIA GPU yet.

## Health, next steps and techniques form one loop

Health graded the model, next steps measured it and the catalogue proposed experiments, but none of them read the
others: a failing "Holds on new inputs" did not make the proposer try the families aimed at it. They are now one loop,
and AGENTS.md makes it the rule for every feature:

| Link | Where |
|---|---|
| Each family names the checks it `targets` | `byosm/improve/techniques.yaml`, shown as a Targets column on the Techniques page |
| A check that is not ok lists the families aimed at it ("Try ...") | `with_techniques` in `byosm/improve/techniques.py`, Health card |
| The bandit draws a family aimed at a failing check with 1.5 times its score | `BLOCKER_BOOST` in `byosm/improve/bandit.py`; each proposal records its `aims` |
| "Find more experiments" says which failing checks it aims at first, and with which families | `_aims` in `byosm/next_steps.py` |
| The proposer and the lab author read the health checks and the next steps, with the goal of raising the primary score and clearing what is not ok | `diagnose` adds `health` and `next_steps`; prompts in `analyst.py` and `lab/author.py` |

A new check, "Tested on real inputs" (`byosm/evals/real.py`), joins Health: a classifier passes once it is scored on a
public set of real inputs it never trained on (a suite of real data other than its training sources, or a real primary
benchmark), graded by the drop from its own test score with the same 0.05 and 0.15 bounds as the other drops
(`DROP_OK`, `DROP_WARN`, now in `byosm/health_score.py`). Until then it is unknown, which blocks readiness, and its
step adds the public benchmarks that fit or scores the ones added; that step replaces the separate public benchmark
step, and Measure everything lists it as "Real inputs". Check titles live in one map, `CHECK_TITLES`, and the
"Holds on new inputs" check moved to `byosm/evals/robust.py`.

## The local app can reload its own code

Every backend change needed the owner to stop and start `byosm app`. `byosm app --reload` now serves the app through
uvicorn's reloader, which restarts it when a file under `byosm/` changes. The command hands its settings (workspace,
token, hosts, parallel jobs, port) to the worker through one environment variable, and `byosm/app/factory.py` builds
the app from them. Running jobs are detached processes, so a reload leaves them running; the workspace lock records the
app's process id, and the reloader stops the old worker before the new one claims the lock. In the same mode `vite build --watch` keeps the UI the
app serves in step with `ui/src` (`watch` in `byosm/app/ui_build.py`), so port 8765 shows UI changes after a page
refresh. Reloading is for local development: a deployed app runs without it.

## A proposed technique goes to the labs that need it, and proposals are grouped by day

Sending an accepted technique to the lab asked the owner to pick a specialist, against the rule that the app fills
every setting, and it offered text classifiers even for a forecasting technique the lab cannot write yet. "Try it"
(`POST /api/scout/<id>/try`, `chosen` in `byosm/scout/proposals.py`) now sends it to the labs of the two promoted
classifiers of a kind it fits with the lowest health score, records where it went (`sent`), and refuses a kind the lab
does not write code for, which the card says instead of offering the button. Proposals are grouped under the day the
scout found them, newest first, and each shows the date of its paper (`published`, read from arXiv and Hugging Face).

## Catalogue families carry dates

Every family in `byosm/improve/techniques.yaml` records the day it joined the catalogue (`added`), and the Techniques
page shows it next to each evidence link's publication month. The month comes from the arXiv identifier itself
(`2006.05987` is June 2020; `published` in `byosm/improve/techniques.py`), so no lookup is needed; other links show no
date. Proposed techniques show the paper's date and the day the scout found them.

## From a paper to a default technique

A paper the scout found now has a path all the way into the pipeline, with the owner approving at each step that runs
or ships code:

| Step | What happens | Where |
|---|---|---|
| Read | "Try it" passes the paper's link; the lab reads its method and training sections from arXiv's HTML version, cached per paper, and writes code that follows them | `byosm/scout/fulltext.py`, `lab write --paper` |
| Retry | a run that fails or does not win writes a revision from the error or the scores (`parent` on the draft), at most two per method; each version waits for approval | `revise` in `byosm/lab/run.py` |
| Spread | a promotion won with a lab method puts it in the workspace's proven methods; the proposer offers that approved code to other specialists of the kind | `byosm/lab/proven.py`, `note_win` in `_record` |
| Ship | after wins on two specialists a patch adds the method as `byosm/lab/methods/<name>.py`, a catalogue family pinned to `train.lab_method=builtin:<name>` and a test; the owner reviews it and runs `git apply` | `write_patch`, Techniques page "From the lab" |

Shipped methods load from the repository and still pass the static checks before they run. Nothing changes the
repository on its own: the patch is a file in `.byosm/lab/patches/`.

## The lab writes code for every specialist on the shared trainer

The lab began with text classifiers. Its hooks act on the shared trainer, so every specialist built on it can use
them: text, audio, image and document classifiers and taggers. The loss hook of a tagger receives one row per token
or frame, with targets of -100 where the trainer ignores a position (`_task_loss` in `byosm/train/trainer.py`), and
the rows hook may reorder, filter or duplicate file inputs but not invent them. The Lab tab, the `lab-method` family
and the scout's "Try it" now cover these kinds. Forecasters train elsewhere and get their own contract; event-risk
specialists come after.

## Every kind gets a picture of how it decides

A score says how good a specialist is; it does not show how it reaches one answer. Each kind gets a view on its Test
tab that draws the decision on the input it just ran, in the style of a forecast chart: one line over the input, a
reference line, and parts of the input that can be switched off to see the line move.

| Kind | Line | Switch off | Where |
|---|---|---|---|
| Text classifier | probability of the answer after each word (word prefixes, at most `MAX_STEPS` points) | a word: the input is read again without it, and the line keeps following the first answer so the two curves compare | `byosm/app/explain.py`, `ReadingView.tsx` |
| Tagger | confidence of the tag given to each word, with each found span shaded and named by its kind | a word: the input is tagged again without it, and the line breaks where the word was | `confidence` in `rt.py`, `TaggingView.tsx` |
| Event risk | the chance, with each feature's signed effect: the change in chance when that feature takes its typical value | a feature: it is scored with the typical value, and the chance shows what it was | `effects` in `byosm/tabular/reasons.py`, `RiskDrivers.tsx` |
| Document classifier | the first page as the model sees it, with a 6 x 6 heatmap (the drop in the answer when one square is covered in grey), and the answer without the picture and without the text | a file: upload a PDF or a scan, kept once by its content hash; the rows arrive one request each | `byosm/app/occlusion.py`, `PageTry.tsx` |
| Forecaster | history, the forecast with its 80% interval and the now line (already on the Test tab) | not yet: univariate forecasters have no covariates to switch | `SeriesTry.tsx` |

Word drivers come from occlusion: the drop in the answer's probability when one word is left out. It costs one
forward pass per word on the exported ONNX bundle (about half a second for a 60-word tool-choice pair on CPU) and
needs no gradients, so it works on any exported text bundle. The dashed line is the decision threshold when the
answer is the positive label. Event-risk effects are one feature at a time, so they do not add up to the chance:
features interact, and switching one off changes the effect of the others. A toggle sends the row the first lookup
returned, so it scores again without reading the register again.

A fused document model spreads its evidence over the page: covering one square of 36 moves a confident answer by a
fraction of a point, while removing the whole picture can flip it (an RVL-CDIP invoice became a scientific report,
0.963 to 0.022, and kept its label without the text). So the page view shows both: the squares say where, the two
"without" rows say which view the answer rests on. The board agrees: the text-only doctype scored 0.742 on test
against 0.948 for the fused one.

Every view reads the same exported bundle through one cache keyed by the manifest's modification time
(`byosm/app/bundles.py`), shared with Predict, so a new export is picked up without a restart.

## The lab writes training methods for forecasters too

A forecasting technique from the scout was a dead end: the lab only wrote code for text classifiers. A lab method now
belongs to a specialist or to a forecaster domain, and each kind has its own contract behind the same gate.

| | Text classifier | Forecaster |
|---|---|---|
| Drafts live in | the specialist's reports, `lab:<id>` | workspace reports `forecaster:<domain>:lab:<id>` (`Forecaster` in `byosm/lab/owners.py`) |
| Hooks | `rows(train, rng)`, `loss(logits, targets, weight, step)`, `after_step(model, step, total)` | `window(values, context, rng)`, `loss(pred, target, known, levels, step)`, `after_step(model, step, total)` |

`byosm/lab/contract.py` keeps one `Contract` per kind: the hooks, their descriptions, what stays fixed, the model the
method trains and the goal the author is given. Everything else is shared once: the allowed imports, the blocked calls
and attributes, the safe builtins, the opening paragraph, the rules paragraph (now written from the allow-list, so the
text and the check cannot drift apart) and `after_step`. The AST check (`problems(code, kind)`) asks for at least one
hook of that kind, and the loader returns only that kind's hooks. A name resolves to a specialist first, then to a
forecaster domain that has run (`owner`); the kind comes from the owner (`kind_of`) and is recorded on every draft. The
Lab routes, `byosm lab write|ls|show|approve|run` and the approval of the exact code hash work the same for both.
Listing workspace reports by prefix is one helper now (`workspace_reports`, with `LIKE` escaping, since dataset names
such as `car_parts_with_missing` contain `_`), shared by the forecaster board and the event-risk forecasts.

**Training a forecaster with a method.** The hooks act in `train` and `batch` (`byosm/series/patch_net.py`): `window`
gets every training window after the recipe's augmentations and before patches are hidden and the window is scaled,
with `values[:context]` the past and the rest the future up to the longest horizon (NaN past the dataset's own), and
must return as many values as it was given; `loss` replaces the pinball loss and gets the quantile levels as a tensor;
`after_step` runs after every optimiser step. Steps count from 0, as in the classifier trainer. A version option,
`fit.lab_method` (the draft id), turns a method on. `candidates` (`byosm/series/fit.py`) gives the patch transformers,
from scratch and pretrained, a `Method` (the owner and the id) and names them `patch transformer, lab <id>`, so
`refreshed` sees exactly those models change: they are trained again at the promoted effort, and boosting, the rivals
and the rest are only scored again on the same calib and test windows. `byosm lab run <domain> <id>` runs that round
(`try_lab_method` in `byosm/series/run.py` on top of `apply_technique`), so the version it adds is promoted only on a
strictly lower calib MASE, like a technique round. The draft gets the version, whether it was promoted and its calib
MASE, or the error. A method the promoted version already trains with is not run again. The hooks are loaded when
training starts, never stored on the model, so a saved version pickles only the reference. The option stays on in the
versions built from a promoted one, so Keep training goes on training the patch transformers with the method; if the
method is rejected later, those rounds stop with "not approved" instead of training without it. The catalogue's
`lab-method` entry now covers `fit.lab_method` and targets `rival`. Not measured: no lab method has run on a GIFT-Eval
forecaster yet; the tests use fakes and a tiny patch transformer on the CPU.

**Writing a forecaster's method.** `byosm lab write <domain>` (and "Write a method" on the forecaster's Lab tab) gives
the `lab` role the forecaster contract and `diagnose` from `byosm/series/diagnosis.py`: the datasets, the promoted
version with its options, the best rival on the datasets both were scored on, calib MASE, test MASE and CRPS per
dataset for every model the promoted version was scored with, the health checks, the next steps and every version so
far, then the earlier methods with their outcome or error and the owner's ask. Its goal is a lower calib MASE than the
promoted version, to beat the rival and clear the checks that are not ok. The classifier prompt now reads its next
steps as well. To close the loop, a forecaster's "Beats the best rival" check, when it is not ok, names its fix: the
technique Keep training tries next ("Try Multi-rate forecasting", `series run <domain> --until-rival`) or, once none is
left at the promoted effort, "Try Lab method" (`lab write <domain>`). `next_steps` in `byosm/series/steps.py` lists
those fixes and the pretraining v2 step, which moved there from the measure plan, so both read one rule.

**Where a proposal goes.** "Try it" on a forecasting technique (`kinds` holds `series`) now sends it to the labs of the
promoted forecasters, not archived, lowest health score first, and the two lowest across classifiers and forecasters
are picked when a proposal fits both (`chosen` in `byosm/scout/routing.py`, which also holds `lab_fits`, `ask` and
`mark_sent`, moved out of `proposals.py`). The lab kinds are `text`, `audio`, `image`, `document` and `series`, and the
scout's list says per proposal whether the lab can try it (`lab`), so the page and the route read one rule; event
risk is still refused with a 422.

**On the page.** A forecaster has a Lab tab after Versions, the same component as a specialist's with the kind's rule
in its tooltip and its score named calib MASE; the sure lead badge stays a classifier's. The Versions table's Options
column shows `lab <id>` for a version trained with a method. The Techniques page offers Try it on forecasting
techniques and says "No lab for ... yet" only for kinds the lab does not write. Checked in the app on `energy`: the
writer read its board and wrote a method aimed at solar/H's calib/test gap, which waits for approval.

**Next steps and revisions.** A forecaster's page shows Next steps beside Health, the same card as a specialist's,
read from `byosm/series/steps.py` through `/api/next-steps/<name>`; each fix says which check it clears. Its "Ask the
LLM" writes a lab method, and there is no "Do all" because the ready loop drives specialists only. A forecaster's
method that fails or loses gets a revision for approval like any other; one its promoted version already trains with
is refused (`RefusedError`) without an error on the draft or a revision.

## A manual experiment that repeats another attempt is refused

The ready loop's LLM proposed near-miss negatives for tool-choice-held and created `v2` with them a minute before a
manual `byosm exp create ... --set data.pairs.near_misses=[...]` made the same change as `near-miss`: the GPU then
trained one idea twice. `improve --apply` already refused a proposal whose changes an earlier attempt makes;
`exp create` did not. Both now go through `same_attempt` in `byosm/improve/proposals.py`, which compares the
normalised changes (`proposal_key`). A manual experiment is refused when a different attempt makes the same changes,
with the command that builds that one; recreating the same experiment name is allowed, so a next step can be run
again, and `--force` trains the changes again on purpose.

## A lab method waits for a person wherever work is listed

A method the lab wrote was only visible on the Lab tab: the Techniques page said "Sent to energy, operations" and
nothing else, and an approved method waited for someone to find its Run button. Now `byosm/lab/steps.py` turns the
lab into steps, for specialists and forecasters alike:

| Draft | Step | Who acts |
|---|---|---|
| pending, passes its checks | "Approve lab method X", no command, `approve` set to its id | a person, with Approve or Reject |
| approved, not run | "Run lab method X", `byosm lab run <name> <id>` | the ready loop, "Do all", or a person |

The approval steps show in a specialist's or forecaster's Next steps, pinned at the top of Home's Next actions
(`/api/lab/pending`, one request for the whole workspace) and on the technique that produced them (each proposal lists
its methods by paper, linked to `#/specialists/<name>/lab`). Every place uses the same Approve and Reject buttons
(`MethodActions.tsx`). Approval stays a person's decision: an approval step has no command, the ready loop and the
bulk buttons take only steps with one, and the job runner refuses `lab approve` (`PERSON_ONLY`), so neither an LLM
nor a queued job can approve code.

A forecaster's promoted version trained before models were saved has nothing to refit. A lab run then trains all of
that version's models again with the method, at the same effort, and promotes the result only on a lower calib
MASE; the same function backs Keep training (`_fit_version` in `byosm/series/run.py`). A run that is refused (no
promoted version, or the method already in use) records why on the draft, shown on its card and in its step.

## A latency timed on a busy machine does not count

CPU per input is the median of single-input runs taken while the suite is scored, whatever else the machine does.
tool-choice-held's `v2-r1` showed 670.5 ms against 110.5 ms for `v1`, same size, scored while another training and a
test run held the CPU: a number that would fail "Fast enough" and line up a distillation the model does not need.
Scoring now records the load per core next to each latency (`latency_load`, from `load_per_core` in
`byosm/machine.py`, the same reader as the machine page). Above `BUSY_LOAD` (0.3 per core) the timing is not fair:
`fair_ms` in `byosm/evals/scoring.py` returns nothing for it, so the speed check says unknown instead of slow and the
speed rule in promotion ignores it, and the board marks it busy. Timings scored before this have no load and keep
their value.

A busy timing closes the loop like any other check: the speed check's fix is `byosm time <name>`, shown as "Time it
again" in Next steps and run by the ready loop. It waits until the load is back under `BUSY_LOAD` (polling every
`QUIET_POLL_S`, for at most `QUIET_WAIT_S`), times each experiment whose primary timing was busy on the primary suite,
and saves only the speed fields, keeping every score (`byosm/evals/timing.py`). A machine that stays busy fails the
job instead of saving another unfair number; `--exp` times any experiment again, including ones timed before the load
was recorded.

The threshold started at 1.0 per core and was too loose: timed again back to back at 0.71 to 0.91 load per core,
tool-choice-held's `v1`, `v2-r1` and `near-miss` took 625, 677 and 596 ms, so the near-miss data did not slow the
model, but `v1` had been timed at 110.5 ms on a quieter machine: the experiment predictor runs PyTorch on every core,
and a load of 0.7 per core already made it about 5.7 times slower. At 0.3 those timings count as busy and wait to be
taken again.

## Two models compared on the same rows

"Is this run better than that one?" had no answer on the page: the board shows each model's own interval, and two
overlapping intervals say nothing about the difference. `byosm/evals/versus.py` runs the paired bootstrap the
promotion already uses (`compare`) for two models on every suite both were scored on, on calib and on test, and the
Board tab's "Compare two models" card draws the test difference per suite as a forest plot with a table of the calib
lead, a "sure lead" badge at the same 0.95 the promotion requires, and the test lead. The forest plot is the same
chart as the catalogue's calib effect (`ForestChart.tsx`).

On tool-choice-held it settled a question the separate intervals could not: the screening rung `v2-r1` (one epoch)
scored 0.967 on test against 0.958 for the full three-epoch near-miss run, but on calib the lead is -0.001
[-0.011, +0.008] with p 0.40, so the shorter schedule is not better; both lead `v1` by 0.046 on calib.

## A technique leaves Proposed once it has been tried

Proposed techniques is the scout's inbox: a paper's idea waits there to be accepted and sent to a lab. Approving a
method's code says nothing about whether the idea works, so the move is keyed on its methods having run: a proposal
with at least one method that reached a result leaves the card and joins the catalogue as a `lab` row
(`byosm/scout/tried.py`), with its paper, the specialists it was tried on and one outcome per method, linked to that
specialist's Lab tab. A classifier's win is a sure lead; a forecaster's is a promotion on a lower calib MASE, which is
not a sure lead, so it is shown as "promoted" and does not count in the row's sure leads. The summary cards and the
two family charts count the catalogue's own families only: a classifier's lab experiments are already counted under
"Lab method". A technique becomes a default only through the proven registry (sure leads on two specialists), so a
forecaster's method cannot become one until forecaster promotion asks for a sure lead.

## Seed runs are scored on the primary suite before the vote

A promotion vote replicates the candidate and the promoted recipe on other seeds and reads only their calib score on
the primary suite (`maybe_promote`, `seed_sd`, the seeds report). Each replica used to be scored on every suite before
the vote, and on the tool-choice specialist the four other suites took about 2.7 of a replica's 3.2 scoring hours (job
0efcab008d: near-miss-s1 scored builtin in 34 minutes, then bfcl, bfcl-live and when2call in 159). Replicas are now
scored on the primary suite inside `build(..., suites=[primary])`, the vote is logged, and `score_rest` scores the
replicas on the other suites afterwards in the same job, so the spread of seeds on out-of-domain suites (bfcl-live
calib 0.69 against 0.62 for two seeds of near-miss) is still measured, just after the decision. `byosm seeds` does the
same. The soup is still scored on every suite before shipping, because a shipped soup is the promoted model.

## Scoring reports how far it has got

Scoring a 395M encoder on CPU takes hours on the bigger suites (when2call: 8797 test rows, 82 minutes), and the only
event was `scored` at the end of a suite, so a job showed "Scored v4-r1 on bfcl" for over an hour while it scored
bfcl-live. `_collect` now scores a local model's rows in chunks of `SCORE_CHUNK` (256, a multiple of every predictor's
batch, so the batches and the scores are unchanged) and emits `scoring` with the suite, the split and rows done of
total after each one; the job's current step reads "Scoring v4-r1 on bfcl-live test, 1024/5253" with a time left.
Predictors that take whole rows (an LLM rival, the teacher panel) are not chunked: they batch and parallelise their
own calls and already report each call. The specialist page's Running now card shows the same current step as the
job list (`stepNow`).

## The seed spread is drawn next to the board

A promotion decides on the mean of several seeds of a recipe, but the board showed each seed as an unrelated row, so a
lead and the noise behind it could not be compared by eye. `seed_spread` (`byosm/replicas.py`) returns the clean runs
of every recipe on the primary suite, grouped exactly as the vote groups them (`clean_runs`: the same recipe, the same
suite version, calib held out), with the pooled `seed_sd`. A recipe is shown when it has more than one run, and the
promoted recipe always, so a candidate's seeds sit next to what it must beat. Soups are left out: they are an average
of the seeds, not a seed. The Board tab draws one row per recipe, a dot per seed on calib and a tick at the mean
(`SeedSpread.tsx`); it comes with the specialist detail, so it updates with the board and has no loading state. On the
tool-choice specialist it showed near-miss's two seeds at 0.931 and 0.959, a spread as wide as the leads being voted
on.

## The seeds table counts the recipe as its first seed

`byosm seeds` and a promotion vote train `<recipe>-s1`, `<recipe>-s2` next to the recipe's own run, which is the first
seed (`clean_runs`). The report's seeds table grouped only the runs named `-s<k>`, so it averaged one seed fewer than
the vote, dropped a recipe with one replica (tool-choice near-miss had no row), and counted a soup as a seed, since
`recipe_of` maps a soup to its recipe. `seed_recipe` (`byosm/report/seeds.py`) now names the recipe of a seed run and
leaves soups out, and a recipe that has seed runs joins its own group. The main table still lists the recipe's own run
and the soup and hides only the `-s<k>` replicas off the promoted lineage, so no run disappears from the page.

## Every promotion is logged

The shipped model was one column, `specialists.promoted`, overwritten in place, so nothing said when a specialist's
model changed, from what, why, or which job did it. `Project.promote(exp, why)` is the one place a promotion happens,
and it now adds a `promotions` row (before, after, why, by, at) in the same transaction. `by` is the job id from
`BYOSM_JOB_ID` (moved to `byosm/events.py` next to the other job variable, so the core does not import the GPU
module) or `terminal` for a command typed by hand. The reason is required: the vote passes the same sentence it logs
(calib, seeds, P(better), seed sd, test), and adopting a held-out set, a speed promotion, a distilled student and
`byosm promote` each say what they did. The Runs tab lists them newest first, with the job's name linked. There is no
backfill: the old promotions left no record, and `specialists.updated` also changes when the spec is saved, so a date
taken from it would be a guess. The log starts with this change.

## A suite's score is broken down by kind of request

The tool-choice specialist failed "Holds on new inputs" by 0.45 on When2Call, and nothing on screen said why, although
scoring already stores exact accuracy per category for pair suites (`grouped.per_request`). The detail payload now
passes it as `cases` and the Board tab draws it per suite ("By case"): one bar per model and kind of request with its
95% interval, the models and colours taken from "Every suite" so a model keeps its colour. On When2Call it shows that
the gap is one case: when a required detail is missing and the right answer is to ask for it, every model calls the
tool anyway (near-miss 0.32, v1 0.30), while "cannot answer" rose from 0.66 to 0.77 with near-miss negatives.

The same breakdown closes the loop: `robust_check` adds the weakest kind of request on the hardest suite to its reason
("Weakest kind of request on when2call: request for info, 0.30 exact over 856 requests"), so Health, the next step and
the proposer and lab LLMs, which read the checks, see which case to aim at rather than only the drop. `grouped.cases`
is the one reader of the stored breakdown for both the detail payload and the check.

## Pages load from one read per request and a calm live stream

Measured on a loaded machine (`docs/SPEED.md`): Home fired 108 requests and finished after about 41 s with 6.5 s of
blocked main thread; Specialists took 31 s, and a light page opened next waited 1.2 s for its config because the
server was still working through the previous page's queue.

- **The live stream is applied in batches.** On connect the server sends every job as its own message (244 here).
  Each message used to re-render the app and change eleven hand-written "finished jobs" counts, and every page
  refetched on each change: 57 health, 17 specialist and 10 next-step requests for one Home load. `LiveProvider` now
  applies messages every `FLUSH_MS` and counts, per command, the jobs it saw running that have ended; every page
  reads it through `useFinished`, and forecasts and open data read only their own command's count. They used to
  count finished jobs of their kind, which jumps from 0 when the job list first arrives, so each fetched twice on
  load (`/api/forecasts` is 225 KB).
- **Per-item requests run four at a time and stop when the page is left** (`useEach`), so health and next steps for
  24 specialists no longer queue ahead of whatever page comes next.
- **A read request sees one board.** Health read the board 11 to 22 times, each decoding every item's units (95% of
  the stored bytes). GET requests run inside a `snapshot` (`byosm/reads.py`, `ReadSnapshot` middleware) in which the
  first board read is kept for the rest of the request. Jobs and writes never run in one, so they always read fresh
  scores. The specialist list reads boards without units (`board(units=False)`, the database drops the key), since
  it shows scores only.
- **A classifier no longer loads every forecaster** to learn it is not one: `report_of` reads its one report.

Specialists asks for next steps per card like Home, and a card re-renders only when its own data changes
(`memo`). It asks only for the cards on screen, in the order they are drawn, and `useEach` fetches a key once per
refresh, so switching tabs and back fetches nothing again. The command palette loads the specialist list when it
opens instead of on every page load. Together that drops 56 requests to 47 per load, about 0.8 s of server time. After: Home has all its data at 7.2 s with 0.83 s blocked, Specialists at 6.4 s with 0.26 s, and a page
opened next answers in about 20 ms. One health request takes 0.1 to 0.3 s instead of about 0.7 s.

While planning this, `maybe_promote` turned out to count a missing paired comparison as a sure lead whenever the
promoted model had runs but either side had no units. It now fails closed: with runs on both sides and no paired
comparison, the lead is not sure.

## A read request sees each specialist row once

After the label index, the health and next steps behind one Specialists page still made 3,016 queries in-process:
1,205 specialist id lookups, 412 promoted lookups, 410 spec reads and 372 experiment reads, plus 20,658 JSON decodes.
Every `Project` read now goes through `_once`, the same per-request `snapshot` the board uses: inside a GET request
the id, spec row, promoted model, experiment lists and each experiment row are read once; outside one (jobs, writes)
every read is fresh. The whole page's health and next steps went from 1.14 s to 0.89 s in-process, with half the
function calls.

- **Returned values are copies at the top.** `experiment` returns a new dict and the lists are new lists, so a caller
  that reassigns a field does not change the next read. Nested values (an experiment's `meta`) are shared within the
  request; read paths do not write into them.
- **Reports are not remembered.** Health writes the paired lead it computes into a report during a read, and a
  remembered report would hide that write from the rest of the request.
- **A proposal is read on its own.** Next steps, the technique board, the TPE search and diagnosis loop over every
  attempt for its proposal, and each call used to read the whole experiment row (spec, meta and errors too). `proposal` selects
  only that column. The technique board went from about 0.6 s to 0.35 s of CPU; loading all proposals at once
  instead was rejected because jobs call `proposal` in loops outside a request, where it would read every row per
  call.
- **Published reference tables are read once per request.** The forecasters' cards compared against GIFT-Eval's
  published scores by reading the seasonal naive and both Timestral tables again for every dataset: 159 file reads
  of 3 files per list request, now 3.

## A request missing a required argument does not fit yet

Tool choice's labels said only "this option is one the request should use". The near-miss miner writes requests with
one required argument left out and labels them no fit, but the Haiku judge relabels every mined pair from the label
definitions, and under that wording the tool is the right one: it turned all 797 of them into fits. When2Call scores
the same situation the other way (ask for the argument, call nothing), so the training data taught the opposite of
the benchmark, and "ask for missing info" stayed at 0.26 to 0.34 exact in every run while near-miss lifted "no tool
fits" from 0.66 to 0.75. The other kinds were flipped far less (nearest 26%, same category 10%, removed 7%).

- **The pair labels are defined once** (`byosm/options/situations.py` `DEFINITIONS`) and say that a request which
  leaves out a required argument should not call the option yet. New option specialists get them from `spec_for`;
  tool choice and tool choice held were moved to them. The LLM rivals read the same definitions, so a rival scored
  before the change was judged under the old wording.
- **Data an LLM labels is keyed by the label definitions.** The data key took the definitions into account only for
  synthetic data, so rows the judge relabelled, cleaned or a teacher panel labelled kept their key after a definition
  change and would have been reused with the old labels. `DataPlan.labelled_by_llm` (synthetic, clean labels,
  teachers, near misses) now puts the definitions into the key. That moved the data of 12 experiments (tool choice
  held's near-miss and clean-label recipes, bank panel's teacher panel); their rows and folders were copied once to
  the new keys, and the old copies stay until the jobs that started on the old code have ended.

## The ready loop's budget holds inside a step

The ready loop checked its 6 h budget only between steps, and one step ("Run N experiments") trains a model per
proposal, so tool choice held's loop ran for more than 26 h: the step started inside the budget and screened six
proposals back to back. `until_ready` now runs under `deadline.until`, and `build` calls `ensure_time_left` before it
trains anything, so a step stops before its next model once the budget is spent; the loop logs which model was not
started and ends with "time budget spent". A build outside the ready loop has no deadline. Screening lets that stop
through instead of logging it as a failed run.

## Screening runs are scored on the primary suite only

A screening run (`-r1`, a third of the epochs) is ranked by its primary calib score alone, yet each one was scored on
every suite: on tool choice held that took 1 to 6 h per run, as long as the training. Rungs now build with
`suites=[primary]`; a proposal that wins the screen gets its full training, which is scored on every suite.

## Robustness compares whole requests when both suites count them

"Holds on new inputs" and "Tested on real inputs" took the drop in the objective from the primary suite to the worst
other one. For tool choice the objective is F1 over (request, tool) pairs, whose value depends on the share of pairs
that fit: 57% on the builtin suite, 12% on When2Call, so F1 fell from 0.924 to 0.470 partly by arithmetic. Both checks
now share `worst_drop` (`byosm/evals/drop.py`), which compares exact requests (all offered tools right) when both
suites report them and the objective otherwise, and names the metric in the value: -0.364 exact requests on When2Call
instead of -0.453. The drop is still a fail; it is now one users would see.

The fix the robustness check names used to be Keep training, which writes rows aimed at the promoted model's dev
errors. Tool choice held's dev split has the training distribution and 8 errors, so there is nothing to aim at; with
fewer than `AIMABLE_ERRORS` (20) the check names the proposer instead, which reads this check and the techniques in
the catalogue that target it.

## Promotion can decide on several suites

Ten tool choice held runs land at 0.93 to 0.96 on builtin calib, within seed noise of each other, while the recipes
that top it (v5-r1, v6-r1) are among the worst on When2Call and BFCL-live calib. `selection_suites` in the spec names
the suites a promotion decides on; empty means the primary alone, so every other specialist votes as before. The
value is the mean of the calib scores over those suites (`selection_mean`), leaving out a suite with fewer than
`MIN_CALIB_ITEMS` calib items (`selected`); the sure lead resamples each suite's calib items on its own and averages
the per-suite deltas (`compare_suites`, `bootstrap_suites`), and one suite is the old paired bootstrap exactly. Seed
replicas are built and counted on every selection suite, seed noise is the spread of the mean, and the reason names
each suite's delta. Test stays report-only. Tool choice held selects on builtin, BFCL, BFCL-live and When2Call; the
LLM-written suite has 8 calib items. `byosm vote NAME EXP` reruns the decision for a trained experiment, which jobs
started on older code need. The plan and the second half (hard-case rounds) are in `docs/HARD_CASES.md`.

## An LLM rival's cached score depends on its prompt

A rival's cached score is reused while the suite's rows, the label names, the decision rule and the rival's identity
stay the same. For an LLM rival the identity was its model configuration only, but its prompt also carries the task
summary, the input description and the label definitions (`task_context`), so after the tool choice definitions
changed its Claude Haiku score would have been reused as if nothing had happened. `rival_identity` adds a hash of
the task context for LLM rivals, at the one place the cache check reads it (`byosm/evals/board.py` `_identity`).
The 23 cached LLM rival scores whose task text has not changed since they were scored were stamped once with today's
hash, so they are not rescored; plain tool choice's two Claude Haiku scores were left unstamped and are scored again
on its next eval, under the new definitions. Their stored identity also says `claude-opus-5-5`, though the rival is
named and now configured as Claude Haiku: the role most likely fell back to an older default when it was scored, so
the rescore is the first Haiku score there.

The paired lead Health saves (`_lead`) was keyed by the promoted name, the rival's name and the suite fingerprint,
which is the same for every predictor on a suite, so a rival scored again as another model, or a promoted model
retrained under its own name, kept its old lead. The key now carries both identities; every saved lead is computed
once more on the next Health read.

## Spend is summed in the database

The Costs page priced every LLM call one by one: 27,000 call events read row by row, four JSON fields decoded
each, then added up in Python (0.85 s). `usage` now asks the database for one row per job, model and role with the
call count and the summed tokens (261 rows here) and prices each group once, which gives the same totals because a
model's price per token is fixed (0.18 s). Calls that are not done are filtered in the query, and a group with no
price counts all its calls as unpriced.

## Label counts come from an index

The data check in Health counts each label in the promoted model's training rows. For topic that is 90,000 rows,
and SQLite parsed every row's JSON to find its label: 0.43 s of a 0.98 s health request, plus the same again in next
steps. `rows_label` indexes the label next to the row's owner, collection and split, so the count reads only the
index (topic's summary went from about 0.5 s to 0.011 s, health plus next steps from 1.3 s to 0.1 s).

- **The query and the index share one expression.** SQLite uses an index on an expression only when the query spells
  it the same way, and a JSON path sent as a bound parameter does not count: with it the count still took 0.28 s.
  `JsonField` writes the key into the SQL (a key that is not an identifier is refused) and renders
  `json_quote(json_extract(data, '$.label'))` on SQLite and `(data -> 'label')` on Postgres. The Postgres form is
  checked as compiled SQL only; it has not been run against a Postgres server here.
- **Declared indexes are created when the engine starts.** `create_all` creates missing tables but not a new index on
  a table that exists, so the engine also runs `CREATE INDEX IF NOT EXISTS` for every index the models declare. On
  this 1.2 million row database the first start built it in 4 to 10 s; later starts find it and do nothing.
- **Sizes alone are counted, not labelled.** The Datasets page and the training log show only how many rows each
  split has, so they call `Dataset.sizes`, one count per split on the lookup index, instead of tallying every label
  (the Datasets list went from 0.21 s to 0.09 s with the same output).
