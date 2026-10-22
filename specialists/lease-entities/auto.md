# lease-entities: autopilot report

**Result:** v1: **did not win** (time budget: 0.7h left < ~5.3h per experiment, 5.26 h).

- selection (`builtin` calib split): 0.6316 span_f1 vs best baseline hf:Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 0.3704
- held-out test: 0.5137 (95% CI 0.3892-0.6509) vs hf:Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 0.3009: delta +0.2128, P(better) 1.00

## Board

| | params | builtin |
|---|---|---|
| v1 * | 70.69M | 0.5137 |
| hf:h2oai/deberta_finetuned_pii | 138.69M | 0.3556 |
| hf:Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 | 183.92M | 0.3009 |

## Decisions

- `2026-10-22 00:47:54` **define** → text/frame, 7 labels, backbone hf:microsoft/deberta-v3-xsmall: compiled from the prompt; objective span_f1 >= 0.93
- `2026-10-22 00:48:56` **discover:baseline** → hf:Isotonic/deberta-v3-base_finetuned_ai4privacy_v2: token-classification model, 35398 downloads, covers 71% of our labels
- `2026-10-22 00:48:56` **discover:baseline** → hf:h2oai/deberta_finetuned_pii: token-classification model, 7909 downloads, covers 71% of our labels
- `2026-10-22 10:12:56` **build** → v1: builtin span_f1 0.6316; best baseline ('hf:Isotonic/deberta-v3-base_finetuned_ai4privacy_v2', 0.37037037037037035)
- `2026-10-22 10:12:56` **stop** → time budget: 0.7h left < ~5.3h per experiment: 
- `2026-10-22 10:12:56` **distill** → None: skipped: it did not win, so there is no winner to shrink
- `2026-10-22 10:13:03` **overfit** → warn: Some signs of overfitting: read the findings.
- `2026-10-22 10:13:15` **export** → specialists/lease-entities/export: torch-free ONNX bundle (encode / predict / serve); local only, it did not win
