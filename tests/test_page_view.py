import pytest
from fastapi.testclient import TestClient
from PIL import Image

from byosm.app import bundles, create_app, occlusion
from byosm.app.occlusion import GRID, occlusion_row, page, save_tried, tried_path

AUTH = {"Authorization": "Bearer t"}
SIZE = 60


class Model:
    labels = ["form", "letter"]

    def __init__(self, modality="document"):
        self.m = {"modality": modality, "task_type": "classify", "frontend": {}}

    def predict(self, pages):
        covered = pages[0][1].getpixel((1, 1)) == occlusion.COVER
        p = 0.3 if covered else 0.9
        return {"label": "letter" if p > 0.5 else "form", "probs": {"form": round(1 - p, 4), "letter": p}}


@pytest.fixture
def scan(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path))
    monkeypatch.setattr(
        occlusion, "_pages", lambda model, path: [("Dear Sir", Image.new("RGB", (SIZE, SIZE), "white"))]
    )
    return save_tried(b"scan bytes", "page.png")


def test_a_tried_file_is_kept_once_by_content_and_only_its_own_name_resolves(scan):
    assert save_tried(b"scan bytes", "other.png") == scan
    assert tried_path(scan).read_bytes() == b"scan bytes"
    with pytest.raises(ValueError):
        save_tried(b"x", "notes.exe")
    with pytest.raises(ValueError):
        tried_path("../" + scan)
    with pytest.raises(FileNotFoundError):
        tried_path("0123456789abcdef.png")


def test_covering_the_square_the_model_reads_is_the_only_drop(scan):
    read = page(Model(), tried_path(scan))
    assert read["label"] == "letter" and read["grid"] == GRID and read["page"].startswith("data:image/png;base64,")
    assert occlusion_row(Model(), tried_path(scan), 0, "letter") == [0.6] + [0.0] * (GRID - 1)
    assert occlusion_row(Model(), tried_path(scan), 1, "letter") == [0.0] * GRID
    with pytest.raises(ValueError):
        occlusion_row(Model(), tried_path(scan), GRID, "letter")


def test_the_page_routes_serve_document_classifiers_only(ws, scan, monkeypatch):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        url = "/api/specialists/doc"
        assert c.post(f"{url}/page", json={"file": scan}, headers=AUTH).status_code == 404
        monkeypatch.setattr(bundles, "bundle", lambda ws, name: Model())
        assert (
            c.post(f"{url}/tried", files={"file": ("p.png", b"more bytes")}, headers=AUTH)
            .json()["file"]
            .endswith(".png")
        )
        assert c.post(f"{url}/page", json={"file": scan}, headers=AUTH).json()["label"] == "letter"
        assert c.post(f"{url}/page/0", json={"file": scan, "label": "letter"}, headers=AUTH).json()["drops"][0] == 0.6
        assert c.post(f"{url}/page/0", json={"file": scan, "label": "memo"}, headers=AUTH).status_code == 422
        monkeypatch.setattr(bundles, "bundle", lambda ws, name: Model("text"))
        assert c.post(f"{url}/page", json={"file": scan}, headers=AUTH).status_code == 422
