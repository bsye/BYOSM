MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"
DINOV2_SMALL = "hf:facebook/dinov2-small"
TINY_CRNN = "tiny-crnn"
TINY_CNN = "tiny-cnn"
DOCUMENT_ROUTES = "doc"

BACKBONES: dict[str, list[dict]] = {
    "text": [
        {"id": "hf:google/bert_uncased_L-4_H-256_A-4", "params_m": 11, "langs": "en", "note": "tiny BERT, ~1ms on CPU"},
        {
            "id": MINILM,
            "params_m": 22,
            "langs": "en",
            "note": "strong small English encoder",
        },
        {
            "id": "hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "params_m": 118,
            "langs": "50+",
            "note": "small multilingual (most params are embeddings)",
        },
        {
            "id": "hf:distilbert/distilbert-base-cased",
            "params_m": 65,
            "langs": "en",
            "note": "cased: keeps capitalisation, which names and token tagging need",
        },
        {
            "id": "hf:microsoft/deberta-v3-xsmall",
            "params_m": 71,
            "langs": "en",
            "note": "strongest small English encoder for token tagging (most params are its 128k vocabulary)",
        },
        {"id": "hf:distilbert-base-multilingual-cased", "params_m": 135, "langs": "104", "note": "distilled mBERT"},
        {"id": "hf:google-bert/bert-base-multilingual-cased", "params_m": 178, "langs": "104", "note": "mBERT"},
        {
            "id": "hf:answerdotai/ModernBERT-base",
            "params_m": 150,
            "langs": "en+code",
            "note": "long context (8k), English",
        },
    ],
    "audio": [
        {
            "id": TINY_CRNN,
            "params_m": 0.2,
            "langs": "any",
            "note": "log-mel conv+GRU, causal, 10ms frames, streaming-ready; trained from scratch (JEPA pretraining helps a lot)",
        },
        {
            "id": "hf:facebook/wav2vec2-base",
            "params_m": 95,
            "langs": "en-pretrained",
            "note": "20ms frames, strong speech features, heavy",
        },
    ],
    "document": [
        {
            "id": DOCUMENT_ROUTES,
            "params_m": 44,
            "langs": "en",
            "note": "PDF/scan pages: text route (MiniLM on text layer/OCR) + visual route (DINOv2-small on the rendered page), fused per page, attention over pages; see `document:` in spec",
        },
    ],
    "image": [
        {"id": TINY_CNN, "params_m": 0.4, "langs": "-", "note": "small CNN from scratch, <1ms"},
        {
            "id": DINOV2_SMALL,
            "params_m": 22,
            "langs": "-",
            "note": "DINOv2 ViT-S/14, excellent frozen features",
        },
    ],
}

DEFAULT_BACKBONE: dict[str, str] = {"text": MINILM, "audio": TINY_CRNN, "document": DOCUMENT_ROUTES, "image": TINY_CNN}
