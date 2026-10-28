from types import SimpleNamespace

from binary_specs import binary

from byosm.data.hf_source.cards import UNKNOWN_LICENCE, licence
from byosm.hub.facts import suite_dataset
from byosm.hub.sections import caveats
from byosm.spec import EvalSuite

BFCL = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def test_a_public_suite_is_credited_with_its_dataset_and_licence():
    public = EvalSuite(name="bfcl", kind="local", path="bfcl.jsonl", dataset=BFCL, licence="apache-2.0")
    spec = binary("t", "no_fit", "fits").model_copy(update={"evals": [EvalSuite(name="builtin", primary=True), public]})
    assert suite_dataset(spec, "bfcl") == (BFCL, BFCL)
    assert suite_dataset(spec, "builtin") == ("t-builtin", "byosm/t-builtin")
    facts = SimpleNamespace(spec=spec, base=spec, nc=[], results={"bfcl": {"n_test": 3}})
    assert f"`bfcl` from `{BFCL}` (apache-2.0)" in caveats(facts)


def test_a_licence_is_read_from_the_dataset_card(monkeypatch):
    import huggingface_hub

    class Hub:
        def dataset_info(self, dataset):
            return SimpleNamespace(card_data={"license": "cc-by-4.0"} if dataset == "with" else None)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    assert licence("with") == "cc-by-4.0" and licence("without") == UNKNOWN_LICENCE
