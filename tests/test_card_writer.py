from byosm.hub.card_writer import _fill, unsupported


def test_numbers_in_the_prose_must_come_from_the_facts():
    facts = '{"score": 0.8946, "params": 22714754, "rows": 16060, "precision": 0.9466}'
    assert unsupported("It scores 0.895 with 22.7M parameters on 16,060 rows; the LLM reaches 94.7%.", facts) == []
    assert unsupported("It is 3 times faster and scores 0.97.", facts) == [0.97]


def test_every_placeholder_must_appear_once():
    b = {"results": "R", "test_sets": "T", "usage": "U", "reproduce": "P"}
    assert _fill("{{RESULTS}}\n{{TEST_SETS}}\n{{USAGE}}\n{{REPRODUCE}}", b) == "R\nT\nU\nP"
    assert _fill("{{RESULTS}}\n{{USAGE}}", b) is None
