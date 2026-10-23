from test_opendata import CKAN

import byosm.opendata.catalog as catalog
from byosm.opendata import finder, sources

PLAN = {
    "queries": ["company dissolutions"],
    "datasets": [
        {
            "title": "Insolvencies",
            "page": "https://stats/ins",
            "files": [{"name": "ins", "format": "csv", "url": "https://stats/ins.csv"}],
        },
        {
            "title": "Made up",
            "page": "https://nowhere",
            "files": [{"name": "x", "format": "csv", "url": "https://nowhere/x.csv"}],
        },
    ],
    "portals": [{"name": "Open NI", "url": "https://admin.opendatani.gov.uk", "why": "Northern Ireland"}],
}


def _portal_only(url):
    if "opendatani" in url or "gov.ie" in url:
        return CKAN
    raise OSError("not a portal")


def test_a_named_source_is_saved_as_a_portal_when_it_answers_like_one(ws, monkeypatch):
    monkeypatch.setattr(catalog, "_get", _portal_only)
    sources.add(ws, "admin.opendatani.gov.uk/", "Open NI")
    sources.add(ws, "https://www.cso.ie/en/statistics")
    mine = {s["url"]: s["kind"] for s in sources.saved(ws)}
    assert mine == {"https://admin.opendatani.gov.uk": "ckan", "https://www.cso.ie/en/statistics": "page"}
    assert [p["url"] for p in sources.portals(ws)] == ["https://admin.opendatani.gov.uk"]
    assert "data.gov.ie" in {s["name"] for s in sources.listed(ws)}
    sources.remove(ws, "https://www.cso.ie/en/statistics")
    assert len(sources.saved(ws)) == 1


def test_a_request_keeps_web_tables_that_download_and_searches_new_portals(ws, monkeypatch):
    monkeypatch.setattr(catalog, "_get", _portal_only)
    monkeypatch.setattr(finder, "_ask", lambda ws, request, log: PLAN)
    monkeypatch.setattr(finder, "_answers", lambda url: "nowhere" not in url)
    out = finder.find(ws, "Irish companies that close", log=lambda m: None)
    assert finder.relevant([{"title": "Aftercare services"}, {"title": "Company Records"}], "Irish companies") == [
        {"title": "Company Records"}
    ]
    titles = [d["title"] for d in out["datasets"]]
    assert titles == ["Insolvencies", "Company Records"]
    assert out["status"] == finder.DONE and out["queries"] == ["company dissolutions"]
    assert [p["name"] for p in sources.portals(ws)] == ["Open NI"]
    assert finder.requests(ws)[0]["request"] == "Irish companies that close"
    assert finder.forget(ws, "Irish companies that close") == []


def test_without_llm_queries_the_request_keywords_are_searched(ws, monkeypatch):
    monkeypatch.setattr(catalog, "_get", _portal_only)
    monkeypatch.setattr(finder, "_ask", lambda ws, request, log: {})
    out = finder.find(ws, "Records of the companies with their owners", log=lambda m: None)
    assert out["queries"] == ["records companies owners"] and [d["title"] for d in out["datasets"]] == [
        "Company Records"
    ]


SOCRATA = {
    "results": [
        {
            "resource": {"id": "4ijn-s7e5", "name": "Food Inspections", "description": "restaurants"},
            "metadata": {"domain": "data.city.org"},
        }
    ]
}


def test_a_socrata_portal_is_found_from_any_of_its_pages_and_searched(ws, monkeypatch):
    def answer(url):
        if url.startswith("https://data.city.org/api/catalog/v1"):
            return SOCRATA
        raise OSError("no")

    monkeypatch.setattr(catalog, "_get", answer)
    sources.add(ws, "https://data.city.org/stories/food", "City")
    assert sources.saved(ws)[0]["kind"] == "socrata" and sources.saved(ws)[0]["url"] == "https://data.city.org"
    (found,) = catalog.search("food", ["City"], sources.portals(ws))
    assert found["files"][0]["url"] == "https://data.city.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"
