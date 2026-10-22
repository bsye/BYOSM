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

**Rivals.** `llm` is Qwen 3.8 27B, local and zero-shot. `jev` is TypeSafe's Jev judgment model, zero-shot over the same
label definitions, one API request per item; its time column is the API round trip, not local CPU. `hf:` rivals are
public fine-tuned checkpoints.

<p align="center"><a href="../README.md">Tour</a> · <a href="assumptions.md">Goal and bet</a> · <b>Methodology</b> · <a href="results.md">Results</a> · <a href="details.md">Details</a></p>
