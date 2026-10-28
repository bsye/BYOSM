<p align="center"><a href="../README.md">Tour</a> · <a href="assumptions.md">Goal and bet</a> · <b>Methodology</b> · <a href="results.md">Results</a> · <a href="details.md">Details</a></p>

# Methodology

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


## Why the numbers hold up

A small model "beating GPT" is easy to fake. Each of these guards runs in code on every specialist; none is a promise in prose.

| Risk | Guard |
|---|---|
| Grading on your own homework | Rivals (LLMs, Hugging Face models, LiveKit, Silero) go through the same scorer, on the same frozen splits, with thresholds calibrated on the same calib split. |
| Picking the lucky run | Promotion and the autopilot decide on calib only. Test is reported once, with a 95% bootstrap interval. |
| A lead that is noise | Every "ours vs theirs" is a paired bootstrap over the same rows; the health check counts a lead only when 95% of resamples keep it. |
| Memorising the training data | The overfitting check compares train, dev, calib and test, flags a dev curve that peaks and falls, and flags calib over-use. |
| Test rows leaking into training | Near-duplicate rows are grouped before splitting, so paraphrases never straddle train and test; the overfitting check also measures train/test overlap and the score without it. |
| Tiny test sets | Health fails "Precise score" below 100 test rows; a build adopts a bigger held-out set when an experiment brings one. |
| Only good on its own kind of data | "Holds on new inputs" scores it on test sets written separately; a model is not production ready until that is measured. |
| A metric that hides the trade-off | Constrained metrics carry their bound ("recall at 98% precision"), and a rival that never reaches the bound shows what it does reach. |
| Numbers typed by hand | The result tables are regenerated from the results database by `byosm report`; model cards separate what code measured from what an LLM wrote. |

Every number: [Results](results.md).

## Benchmarks

Public and industry-standard where one exists. Every rival is scored by the same code on the same frozen splits.

| specialist | benchmark | metric |
|---|---|---|
| vad | AVA-Speech (`nccratliri/vad-human-ava-speech`) | f1 |
| doctype | RVL-CDIP (`chainyo/rvl-cdip`) | accuracy |
| intent, bank-panel | Banking77 (`mteb/banking77`) | accuracy |
| emotion | `dair-ai/emotion` | accuracy |
| topic | AG News (`fancyzhx/ag_news`) | accuracy |
| ner | CoNLL-2003 (`BramVanroy/conll2003`) | span_f1 |
| assistant, assistant-intent, assistant-args | MASSIVE (`AmazonScience/massive`) | exact frame, accuracy, span_f1 |
| benefits-intent, card-spend | synthetic public data, builtin held-out split | accuracy |
| lease-entities | LLM-written, not a public benchmark | span_f1 |
| turn-completion | builtin held-out split | recall_at_precision |

**Hard public suites.** `byosm suite public <name>` adds the public sets where strong references are below their
ceiling, frozen with their licence, and keeps each one only after its protocol check: `always-positive` must score
what the benchmark itself implies or publishes. None has been scored on a specialist yet.

| suite | for | dataset (licence) | protocol check |
|---|---|---|---|
| `bfcl`, `bfcl-live` | tool choice | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` v3 (Apache-2.0) | calling every tool scores 0 on irrelevance and the label shares elsewhere |
| `when2call` | tool choice | `nvidia/When2Call` multiple-choice test (CC-BY-4.0) | the same |
| `eot-bench` | turn completion | `livekit/eot-bench-data` (CC-BY-4.0) | answering "finished" at once reproduces eot-bench's published silence-timeout cutoffs at 300 and 600 ms |

**Rivals.** `llm` is Qwen 3.8 27B, local and zero-shot. `jev` is TypeSafe's Jev judgment model, zero-shot over the same
label definitions, one API request per item; its time column is the API round trip, not local CPU. `hf:` rivals are
public fine-tuned checkpoints.

<p align="center"><a href="../README.md">Tour</a> · <a href="assumptions.md">Goal and bet</a> · <b>Methodology</b> · <a href="results.md">Results</a> · <a href="details.md">Details</a></p>
