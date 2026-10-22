from byosm.data.hf_source.rows import field, rule_label
from byosm.spec import HFSource

ROW = {"conversations": [{"value": "system"}, {"value": "book a table"}, {"value": "<tool_call>{...}</tool_call>"}]}


def test_a_path_reaches_into_nested_lists_and_dicts():
    assert field(ROW, "conversations.1.value") == "book a table"


def test_rules_label_by_content_then_fall_back_to_the_default():
    rules = [["conversations.2.value", "<tool_call>", "call_tool"]]
    src = HFSource(dataset="a/b", label_rules=rules, label_default="answer")
    assert rule_label(ROW, src) == "call_tool"
    plain = {"conversations": [{"value": "s"}, {"value": "hi"}, {"value": "hello"}]}
    assert rule_label(plain, src) == "answer"
    assert rule_label(plain, src.model_copy(update={"label_default": None})) is None
