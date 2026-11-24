from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.games.poker import screenshot

AUTH = {"Authorization": "Bearer t"}


def seat(name, stack, you=False, dealer=False, said="", cards=True):
    return {
        "name": name,
        "stack": stack,
        "has_cards": cards,
        "is_you": you,
        "has_dealer_button": dealer,
        "last_action": said,
    }


SEEN = {
    "hero_cards": ["6♣", "ad"],
    "board": ["10h", "2c", "Ks"],
    "players": [
        seat("John", 998, said="Call"),
        seat("Mike", 1000, dealer=True),
        seat("George", 999),
        seat("You", 998, you=True),
        seat("Deja", 998, said="Call", cards=False),
    ],
    "pot": 5,
}


class Reading:
    def __init__(self, found):
        self.found, self.images = found, []

    def json(self, system, prompt, schema, images=None):
        self.images.append(images)
        return self.found


def test_a_picture_is_read_into_seats_that_start_after_the_dealer_and_cards_in_one_spelling():
    llm = Reading(SEEN)
    found = screenshot.read(b"png", llm)
    assert llm.images == [[b"png"]]
    assert found["names"] == ["George", "You", "Deja", "John", "Mike"] and found["hero"] == 1
    assert found["cards"] == ["6c", "Ad"] and found["board"] == ["Th", "2c", "Ks"]
    assert found["stacks"] == [999, 998, 998, 998, 1000] and found["pot"] == 5
    assert found["in_hand"] == [True, True, False, True, True] and found["said"] == ["", "", "Call", "Call", ""]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"hero_cards": ["6c"]}, "1 cards in your hand"),
        ({"hero_cards": ["6c", "1x"]}, "not a card"),
        ({"board": ["Th", "2c"]}, "2 cards on the board"),
        ({"board": ["6c", "2c", "Ks"]}, "same card twice"),
        ({"players": SEEN["players"][:1]}, "1 players"),
        ({"players": [{**p, "is_you": False} for p in SEEN["players"]]}, "one seat as yours"),
        ({"players": [{**p, "has_dealer_button": True} for p in SEEN["players"]]}, "one dealer button"),
    ],
)
def test_a_reading_that_cannot_be_a_real_table_is_refused(change, why):
    with pytest.raises(ValueError, match=why):
        screenshot.checked({**SEEN, **change})


def test_the_app_reads_an_uploaded_picture_with_the_local_model(ws, monkeypatch):
    llm = Reading(SEEN)
    monkeypatch.setattr("byosm.llm.recover.local_llm", lambda: llm)
    picture = "data:image/png;base64," + base64.b64encode(b"png bytes").decode()
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        read = c.post("/api/poker/screenshot", headers=AUTH, json={"image": picture})
        assert read.status_code == 200 and read.json()["cards"] == ["6c", "Ad"] and llm.images == [[b"png bytes"]]
        assert c.post("/api/poker/screenshot", headers=AUTH, json={"image": "not base64!"}).status_code == 422
        llm.found = {**SEEN, "hero_cards": []}
        refused = c.post("/api/poker/screenshot", headers=AUTH, json={"image": picture})
        assert refused.status_code == 422 and "cards in your hand" in refused.json()["detail"]
