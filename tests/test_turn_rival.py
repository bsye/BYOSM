import pytest

from byosm.predictors.turn import chat_turns, finished_label
from byosm.spec import Label, TaskSpec


def _spec(names, positive=None):
    labels = [Label(name=n, definition=n) for n in names]
    return TaskSpec(name="t", description="turn", modality="text", labels=labels, positive_label=positive)


def test_a_transcript_becomes_chat_turns_and_plain_text_is_the_user():
    text = "Agent: How can I help?\nUser: I need to change\nmy address"
    assert chat_turns(text) == [
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "I need to change my address"},
    ]
    assert chat_turns("so what I wanted") == [{"role": "user", "content": "so what I wanted"}]


def test_the_finished_label_comes_from_the_spec_or_a_known_name():
    assert finished_label(_spec(["finished", "unfinished"])) == "finished"
    assert finished_label(_spec(["wait", "respond"], positive="respond")) == "respond"
    with pytest.raises(ValueError, match="positive_label"):
        finished_label(_spec(["wait", "respond"]))
    with pytest.raises(ValueError, match="exactly two labels"):
        finished_label(_spec(["a", "b", "c"]))
