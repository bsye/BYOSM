import random

from byosm.data import _group_split3
from byosm.data.store import near_groups


def test_growing_prefixes_of_one_conversation_form_one_group():
    texts = [
        "okay so could you tell me what you think contributes most to air pollution",
        "okay so could you tell me what you think contributes most to air pollution in cities",
        "what time does the store close on sundays",
        "what time does the store close on sundays",
        None,
    ]
    g = near_groups(texts)
    assert g[0] == g[1] and g[2] == g[3] and len({g[0], g[2], g[4]}) == 3


def test_a_split_keeps_each_group_on_one_side_and_leaves_a_test_set():
    rng = random.Random(0)
    convos = [f"conversation {i} about topic {i * 7} with words {i * 13} and more {i * 31}" for i in range(200)]
    items = [(c + suffix, "finished") for c in convos for suffix in ("", " and then", " and then some more")]
    parts = _group_split3(items, rng)
    where = {x.split(" and then")[0]: s for s, rows in parts.items() for x, _ in rows}
    assert all(where[x.split(" and then")[0]] == s for s, rows in parts.items() for x, _ in rows)
    assert len(parts["test"]) >= 30
