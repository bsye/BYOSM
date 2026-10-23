import zipfile

import byosm.opendata.catalog as catalog
from byosm.opendata import cache, library
from byosm.opendata.preview import preview

CKAN = {
    "result": {
        "results": [
            {
                "name": "companies",
                "title": "Company Records",
                "organization": {"title": "CRO"},
                "license_title": "CC BY 4.0",
                "metadata_modified": "2026-10-23T00:00:00",
                "resources": [
                    {"format": "CSV", "url": "https://x/c.zip", "name": "Company Records"},
                    {"format": "PDF", "url": "https://x/a.pdf"},
                ],
            }
        ]
    }
}
EU = {
    "result": {
        "results": [
            {
                "id": "eu1",
                "title": {"ga": "Ainmneacha", "en": "Names"},
                "publisher": {"name": "CSO"},
                "description": {"en": "names"},
                "distributions": [
                    {"format": {"id": "XLSX"}, "download_url": ["https://x/n.xlsx"], "license": {"label": "CC BY 4.0"}},
                    {"format": {"id": "PX"}, "download_url": ["https://x/n.px"]},
                ],
            }
        ]
    }
}


def test_portals_are_searched_into_one_shape_with_tabular_files_only(monkeypatch):
    monkeypatch.setattr(catalog, "_get", lambda url: EU if "europa" in url else CKAN)
    found = catalog.search("companies", ["data.gov.ie", "data.europa.eu"])
    ie, eu = found
    assert (ie["title"], ie["publisher"], [f["format"] for f in ie["files"]]) == ("Company Records", "CRO", ["CSV"])
    assert (eu["title"], eu["licence"], [f["url"] for f in eu["files"]]) == ("Names", "CC BY 4.0", ["https://x/n.xlsx"])


def test_a_zipped_csv_is_downloaded_once_unpacked_and_profiled(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))
    archive = tmp_path / "companies.csv.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("companies.csv", "num,status,closed\n1,Normal,\n2,Dissolved,2009-04-27\n3,Normal,\n")
    url = archive.as_uri()
    info = cache.fetch(url, log=lambda _: None)
    assert info["files"] == ["unzipped/companies.csv"] and cache.fetch(url, log=lambda _: None) == info
    shown = preview(cache.folder(url) / info["files"][0])
    closed = next(c for c in shown["columns"] if c["name"] == "closed")
    assert shown["sampled"] == 3 and closed["missing"] == round(2 / 3, 3) and closed["distinct"] == 1


def test_kept_datasets_are_stored_once_per_file_and_can_be_removed(ws):
    dataset = {
        "source": "data.gov.ie",
        "id": "companies",
        "title": "Company Records",
        "licence": "CC BY 4.0",
        "url": "u",
    }
    file = {"name": "c", "format": "CSV", "url": "https://x/c.zip"}
    library.add(ws, dataset, file)
    kept = library.add(ws, dataset, file)
    assert [k["title"] for k in kept] == ["Company Records"] and kept[0]["cached"] is None
    assert library.remove(ws, file["url"]) == []
