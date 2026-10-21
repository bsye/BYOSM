# emotion: autopilot report

**Result:** v1: **WON** (won: v1 0.9037 beats every baseline (best hf:j-hartmann/emotion-english-distilroberta-base 0.8128) and meets target 0.9, 3.99 h).

- selection (`bench` calib split): 0.9037 accuracy vs best baseline hf:j-hartmann/emotion-english-distilroberta-base 0.8128
- held-out test: 0.8975 (95% CI 0.8827-0.9122) vs hf:j-hartmann/emotion-english-distilroberta-base 0.8378: delta +0.0597, P(better) 1.00

## Board

| | params | builtin | bench |
|---|---|---|---|
| v1 * | 22.72M | – | 0.8975 |
| hf:j-hartmann/emotion-english-distilroberta-base | 82.12M | 0.7426 | 0.8378 |
| hf:SamLowe/roberta-base-go_emotions | 124.67M | 0.5198 | 0.5476 |
| llm | - | 0.4456 | 0.4042 |

## Decisions

- `2026-10-20 22:15:14` **define** → text/classify, 7 labels, backbone hf:sentence-transformers/all-MiniLM-L6-v2: compiled from the prompt; objective accuracy >= 0.9
- `2026-10-20 22:20:53` **discover:benchmark** → dair-ai/emotion: dair-ai/emotion: Directly matches the target: English text classification into 7 emotion categories (joy/love/sadness/anger/fear/surprise + other), with established splits (train/dev/test). (eval on `test`, train on `train`)
- `2026-10-20 22:20:53` **discover:baseline** → llm: the zero-shot LLM is the System-2 bar the specialist must beat
- `2026-10-20 22:20:53` **discover:baseline** → hf:SamLowe/roberta-base-go_emotions: text-classification model, 4672252 downloads, covers 86% of our labels
- `2026-10-20 22:20:53` **discover:baseline** → hf:j-hartmann/emotion-english-distilroberta-base: text-classification model, 558572 downloads, covers 71% of our labels
- `2026-10-21 02:10:29` **build** → v1: bench accuracy 0.9037; best baseline ('hf:j-hartmann/emotion-english-distilroberta-base', 0.8128342245989305)
- `2026-10-21 02:10:29` **stop** → won: v1 0.9037 beats every baseline (best hf:j-hartmann/emotion-english-distilroberta-base 0.8128) and meets target 0.9: 
- `2026-10-21 02:10:30` **distill** → None: not needed: 22.7M <= 25.0M
- `2026-10-21 02:10:38` **export** → specialists/emotion/export: torch-free ONNX bundle (encode / predict / serve)
