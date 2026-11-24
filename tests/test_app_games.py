from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.db.workspace import save_workspace_report
from byosm.games.poker import advisor, table, teacher
from byosm.games.poker.results import ARENA

AUTH = {"Authorization": "Bearer t"}
TAUGHT = {"student": "big-student", "teacher": "big", "model_mb": 5.38, "kl": 0.0727, "top_move": 0.89, "rounds": 0}
WON = {"mbb_per_hand": 5.2, "ci95": 49.3, "hands": 2000000, "seats": 6}


def agreed(name, share):
    said = ["preflop 10 100.0% 70.0% 0.600", f"TOTAL 10 99.6% {share}% 0.635"]
    return {"strategy": name, **table.agreed(said)}


def stored(ws):
    save_workspace_report(ws, "poker-student:big-student", TAUGHT)
    save_workspace_report(ws, "poker-table:big-student:vs:big", {**WON, "strategy": "big-student", "field": "big"})
    save_workspace_report(ws, table.key("big", "caller", False), {**WON, "strategy": "big", "field": "caller"})
    save_workspace_report(ws, table.key("other", "caller", False), {**WON, "strategy": "other", "field": "caller"})
    save_workspace_report(ws, "poker-agreement:big-student", agreed("big-student", 71.6))
    save_workspace_report(ws, "poker-agreement:big", agreed("big", 69.8))
    (teacher.work_dir(ws) / "big-student.pt").write_text("")


def test_a_game_bot_is_listed_as_a_specialist_and_its_page_shows_only_its_own_tables(ws):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert c.get("/api/specialists", headers=AUTH).json() == []
        stored(ws)
        [listed] = c.get("/api/specialists", headers=AUTH).json()
        assert listed["name"] == "big-student" and listed["kind"] == "game" and listed["created"]
        assert listed["score"] == 0.716 and listed["rival"] == {"name": "its teacher", "score": 0.698}
        found = c.get("/api/games/big-student", headers=AUTH).json()
        assert [(r["who"], r["against"]) for r in found["played"]] == [
            ("Student", "copies of the teacher"),
            ("Teacher", "always-call bots"),
        ]
        assert [a["who"] for a in found["agreement"]] == ["Student", "Teacher"]
        assert found["student"] == TAUGHT and found["health"]["score"] > 0
        assert c.get("/api/games/missing", headers=AUTH).status_code == 404
        assert c.get("/api/games/big-student").status_code == 401


class Running:
    def __init__(self, answers):
        self.answers, self.closed = list(answers), False

    def advise(self, hand):
        found = self.answers.pop(0)
        if isinstance(found, Exception):
            raise found
        return found

    def close(self):
        self.closed = True


def test_advice_comes_from_one_engine_kept_running_and_says_why_a_hand_is_refused(ws, monkeypatch):
    advice = {"seat": 3, "street": "preflop", "moves": [{"move": "fold", "probability": 1.0}]}
    started = []

    def start(ws, hands, name, log):
        started.append((name, hands))
        return Running([advice, ValueError("the hand is already over"), OSError("gone")])

    monkeypatch.setattr(advisor.Advisor, "start", start)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        first = c.post("/api/games/big-student/advise", headers=AUTH, json={"hand": "a hand"})
        assert first.status_code == 200 and first.json()["moves"] == advice["moves"] and first.json()["ms"] >= 0
        refused = c.post("/api/games/big-student/advise", headers=AUTH, json={"hand": "over"})
        assert refused.status_code == 422 and "already over" in refused.json()["detail"]
        assert c.post("/api/games/big-student/advise", headers=AUTH, json={"hand": "x"}).status_code == 503
        assert started == [("big-student", None)]
        assert c.post("/api/games/big-student/advise", headers=AUTH, json={"hand": "a hand"}).status_code == 200
        assert len(started) == 2


def test_a_game_bots_health_names_the_step_that_fixes_what_is_not_measured(ws):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        save_workspace_report(ws, "poker-student:big-student", TAUGHT)
        health = c.get("/api/health/big-student", headers=AUTH).json()
        assert not health["ready"] and {x["id"]: x["status"] for x in health["checks"]}["table"] == "unknown"
        steps = c.get("/api/next-steps/big-student", headers=AUTH).json()
        assert steps[0]["args"] == ["poker", "teacher-versus", "big-student", "big"]
        assert ["poker", "teacher-benchmark", "--strategy", "big-student"] in [x["args"] for x in steps]
        stored(ws)
        save_workspace_report(ws, table.key("big-student", "caller", False), {**WON, "mbb_per_hand": 2900.0})
        (teacher.work_dir(ws) / "big-student.abs").write_text("")
        health = c.get("/api/health/big-student", headers=AUTH).json()
        assert not health["ready"] and health["score"] < 100
        [step] = c.get("/api/next-steps/big-student", headers=AUTH).json()
        assert step["args"] == ["poker", "arena-gauntlet", "big-student"]
        for field in ARENA:
            save_workspace_report(ws, f"poker-table:big-student:vs:{field}", {**WON, "mbb_per_hand": 300.0})
        lost = {**WON, "mbb_per_hand": -600.0, "ci95": 60.0}
        save_workspace_report(ws, "poker-table:big-student:vs:nit", lost)
        failed = {x["id"]: x for x in c.get("/api/health/big-student", headers=AUTH).json()["checks"]}["rules"]
        assert failed["status"] == "fail" and "-600.0 ±60.0 mbb/hand against five tight bots" in failed["value"]
        save_workspace_report(ws, "poker-table:big-student:vs:nit", {**WON, "mbb_per_hand": 300.0})
        health = c.get("/api/health/big-student", headers=AUTH).json()
        assert (
            health["ready"]
            and health["score"] == 100
            and c.get("/api/next-steps/big-student", headers=AUTH).json() == []
        )
        wide = {**WON, "strategy": "big-student", "field": "big", "mbb_per_hand": -80.0, "ci95": 70.0, "hands": 1000}
        save_workspace_report(ws, "poker-table:big-student:vs:big", wide)
        [step] = c.get("/api/next-steps/big-student", headers=AUTH).json()
        assert step["args"] == ["poker", "teacher-versus", "big-student", "big", "--hands", "2000"]
