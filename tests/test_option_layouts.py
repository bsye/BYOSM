import json
from typing import get_args

from byosm.options.layouts import RENDERERS, masked, render, required
from byosm.spec.sections import Layout

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {"city": {"type": "str", "description": "the city"}, "units": {"type": "str, optional"}},
}
STOCK = {"name": "get_stock", "description": "A share price.", "parameters": {"ticker": {"type": "str"}}}
REQUEST = "weather in Paris?"


def test_every_layout_the_spec_allows_has_a_renderer():
    assert set(RENDERERS) == set(get_args(Layout))


def test_every_layout_shows_the_request_the_tool_its_parameters_and_the_other_tools_in_its_own_way():
    texts = {layout: render(layout, REQUEST, WEATHER, [WEATHER, STOCK]) for layout in RENDERERS}
    assert len(set(texts.values())) == len(texts)
    for text in texts.values():
        assert all(part in text for part in (REQUEST, "get_weather", "Current weather", "city", "get_stock"))
    assert json.loads(texts["openai"])["tools"][0]["function"]["parameters"]["required"] == ["city"]
    assert texts["prose"].startswith("Offered tool: get_weather.") and texts["prose"].endswith(REQUEST)


def test_a_parameter_is_required_unless_it_is_optional_has_a_default_or_says_so():
    tool = {
        "name": "t",
        "parameters": {
            "a": {},
            "b": {"default": 1},
            "c": {"type": "int, optional"},
            "d": {"required": True, "default": 2},
        },
    }
    assert required(tool) == ["a", "d"]


def test_masking_hides_every_tool_name_but_keeps_the_description_and_parameters():
    option, offered = masked(WEATHER, [STOCK, WEATHER])
    text = render("marked", REQUEST, option, offered)
    assert "get_weather" not in text and "get_stock" not in text
    assert text.endswith("Option: tool_1: Current weather for a city. (city, units)\nOther options: tool_2")
