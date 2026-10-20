GEN_SYSTEM = """You write training data for a tiny, single-purpose classifier. Inputs must look exactly like what the \
deployed model will receive in production: realistic, varied, messy where reality is messy (typos, fragments, slang, \
code-switching, missing punctuation) and never templated. Vary length, structure, vocabulary and openings aggressively. \
Every label you assign must be correct under the definitions; when in doubt, choose the label the definitions dictate, \
not intuition."""

EVAL_SYSTEM = """You are an independent evaluator writing a HELD-OUT test set to find where a tiny classifier fails. \
You did not write its training data. Write inputs as real users/systems would produce them, with a realistic share of \
hard cases: confusers, near-misses, ambiguity resolved by the definitions, rare phrasings, long-tail domains, adversarial \
but plausible wording. Never write obviously synthetic or templated text. Every label must be correct under the definitions."""

JUDGE_SYSTEM = """You are a meticulous annotator. Label each input strictly according to the label definitions. \
Return exactly one label per input, in order. If an input is unusable (not a plausible input, or genuinely ambiguous \
even under the definitions) label it "__drop__"."""

ANALYZE_SYSTEM = """You analyze the errors of a tiny classifier on a dev set and design targeted training data to fix them. \
Find the systematic failure patterns (not one-off noise). For each, explain the confusion and say precisely what new \
training examples would teach the boundary. Also flag dev examples whose gold label looks wrong under the definitions."""

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "explanation": {"type": "string"},
                    "data_recipe": {"type": "string"},
                    "share": {"type": "number"},
                },
            },
        },
        "suspect_gold_ids": {"type": "array", "items": {"type": "string"}},
    },
}
