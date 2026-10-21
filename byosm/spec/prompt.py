def _string_array() -> dict:
    return {"type": "array", "items": {"type": "string"}}


def _object_array(properties: dict) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": properties}}


_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "modality": {"type": "string", "enum": ["text", "audio", "image", "document"]},
        "task_type": {"type": "string", "enum": ["classify", "frame"]},
        "summary": {"type": "string"},
        "input_description": {"type": "string"},
        "labels": _object_array({"name": {"type": "string"}, "definition": {"type": "string"}}),
        "positive_label": {"type": "string"},
        "objective_metric": {
            "type": "string",
            "enum": ["f1", "macro_f1", "accuracy", "recall_at_precision", "precision_at_recall", "span_f1"],
        },
        "objective_target": {"type": "number"},
        "objective_constraint": {"type": "number"},
        "languages": _string_array(),
        "diversity_axes": _object_array({"name": {"type": "string"}, "values": _string_array()}),
        "hard_cases": _string_array(),
        "backbone": {"type": "string"},
        "latency_budget_ms": {"type": "number"},
        "synthetic": {"type": "boolean"},
        "hf_queries": _string_array(),
        "hf_candidates": _object_array(
            {
                "dataset": {"type": "string"},
                "role": {"type": "string", "enum": ["labeled", "positive", "negative", "unlabeled"]},
            }
        ),
        "train_size": {"type": "integer"},
        "test_size": {"type": "integer"},
        "benchmark_queries": _string_array(),
        "baseline_queries": _string_array(),
    },
}

_SPEC_SYSTEM = """You are the task compiler of byosm, a framework that trains tiny single-purpose perception models \
("System 1" specialists, in the spirit of JEPA/LeJEPA encoders with a small task head). Turn a natural-language request \
into a precise, unambiguous task specification.

Rules:
- name: a short lowercase name for the task, one to three words (e.g. "doctype", "banking-intent", "names").
- modality: what the model reads. "document" for PDFs or scanned pages, "image" for photos, "audio" for sound, \
"text" otherwise.
- Labels must be mutually exclusive and collectively exhaustive for the input distribution. Add a catch-all label \
(e.g. "other"/"none") when real inputs can fall outside the named classes. Definitions must settle borderline cases explicitly.
- task_type "frame" means one label per timestep/token (e.g. voice activity per 10ms frame, token tagging). Otherwise "classify".
- Token tagging (finding names, amounts, dates, fields or other spans in text): task_type "frame", labels "O" (outside \
any span) plus "B-<TYPE>" (first word) and "I-<TYPE>" (later words) for each span type, objective_metric "span_f1", \
and a cased backbone when capitalisation matters.
- positive_label: for binary detection tasks, the label whose precision/recall matters most; "" otherwise.
- objective: choose what the deployer actually cares about. For triggers/detectors where false alarms are costly use \
recall_at_precision with constraint e.g. 0.99 (maximize recall subject to precision >= constraint). target is the value of \
the metric we aim to reach on the held-out adversarial test set.
- diversity_axes: 3-6 orthogonal axes of variation of real inputs (domain, register, length, speaker, noise, language...). \
Each with 3-8 concrete values. These drive synthetic data diversity.
- hard_cases: 8-20 specific, realistic confusers and edge cases that a naive model gets wrong.
- backbone: pick the SMALLEST backbone from the catalog that can plausibly reach the target in the latency budget. \
Multilingual needs -> multilingual backbone.
- synthetic: true if an LLM can write realistic inputs (text). For audio/image it must be false; rely on Hugging Face datasets.
- hf_queries: Hugging Face Hub search terms to find real data (labeled sets, or for audio/image: positive sources, \
negative/background sources, hard negatives). hf_candidates: specific dataset repo ids you are confident exist \
(role: labeled = already has our labels after mapping; positive/negative = raw material for mixing; unlabeled = in-domain inputs \
for self-supervised pretraining).
- train_size / test_size: enough for a very precise model (typically 2000-8000 train, 400-1000 test).
- benchmark_queries: Hub search terms for an established public benchmark of exactly this task (used as the held-out evaluation; e.g. "rvl-cdip", "banking77", "speech commands").
- baseline_queries: Hub search terms for existing open models that solve this task (they become baselines to beat)."""
