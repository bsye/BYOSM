from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.data.store import Dataset, record
from byosm.spec import Label, TaskSpec

AUTH = {"Authorization": "Bearer t"}


def test_examples_come_from_the_test_split_per_label_with_published_repos(ws):
    spec = TaskSpec(name="q", description="topic", modality="text", labels=[Label(name=n, definition=n) for n in "ab"])
    root = ws / "specialists" / "q"
    root.mkdir(parents=True)
    spec.save(root / "spec.yaml")
    from byosm.config import Project

    Dataset(Project("q", ws).data_dir_for(spec)).save("test", [record(f"x{i}", "ab"[i % 2], "t") for i in range(10)])
    (root / "hub" / "me__byosm-q").mkdir(parents=True)
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        got = c.get("/api/specialists/q/examples", headers=AUTH).json()
        assert {k: len(v) for k, v in got["examples"].items()} == {"a": 3, "b": 3}
        assert got["repos"] == ["me/byosm-q"]
        assert c.get("/api/specialists/none/examples", headers=AUTH).status_code == 404
