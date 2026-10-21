# topic: autopilot report

**Result:** v1: **did not win** (time budget: -4.0h left < ~6.8h per experiment, 6.96 h).

- selection (`bench` calib split): 0.9141 accuracy vs best baseline llm 0.8000
- held-out test: 0.9027 (95% CI 0.8906-0.9129) vs llm 0.8145: delta +0.0883, P(better) 1.00

## Board

| | params | builtin | bench |
|---|---|---|---|
| v1 * | 22.72M | – | 0.9027 |
| llm | - | 0.8244 | 0.8145 |
| hf:mrm8488/bert-mini-finetuned-age_news-classification | 11.17M | 0.6750 | 0.7305 |
| hf:Maddyhf1/ag-news-topic-classification-distilbert | 66.96M | 0.6696 | 0.7180 |

## Decisions

- `2026-10-21 02:42:16` **define** → text/classify, 5 labels, backbone hf:sentence-transformers/all-MiniLM-L6-v2: compiled from the prompt; objective accuracy >= 0.92
- `2026-10-21 02:47:27` **discover:benchmark** → fancyzhx/ag_news: fancyzhx/ag_news: Matches the classic AG News task (headlines+summary, English news topic classification) with the four core labels (world, sports, business, science/tech) and includes held-out test/train splits; widely used established benchmark. (eval on `test`, train on `train`)
- `2026-10-21 02:47:27` **discover:baseline** → llm: the zero-shot LLM is the System-2 bar the specialist must beat
- `2026-10-21 02:47:27` **discover:baseline** → hf:mrm8488/bert-mini-finetuned-age_news-classification: text-classification model, 1025 downloads, covers 60% of our labels
- `2026-10-21 02:47:27` **discover:baseline** → hf:Maddyhf1/ag-news-topic-classification-distilbert: text-classification model, 27 downloads, covers 60% of our labels
- `2026-10-21 09:35:45` **build** → v1: bench accuracy 0.9141; best baseline ('llm', 0.8)
- `2026-10-21 09:35:45` **stop** → time budget: -4.0h left < ~6.8h per experiment: 
- `2026-10-21 09:35:45` **distill** → None: skipped: it did not win, so there is no winner to shrink
- `2026-10-21 09:35:52` **export** → specialists/topic/export: torch-free ONNX bundle (encode / predict / serve); local only, it did not win
