from byosm.opendata.cache import _name

URL = "https://portal.example/files/table.csv"


def _header(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def test_name_keeps_only_the_basename_of_the_header():
    assert _name(URL, _header("../../../../.zshrc")) == ".zshrc"
    assert _name(URL, _header("../../x")) == "x"
    assert _name(URL, _header("/abs/x")) == "x"


def test_name_skips_dot_names_and_falls_back():
    assert _name(URL, _header("..")) == "table.csv"
    assert _name(URL, _header(".")) == "table.csv"
    assert _name("https://portal.example/a/..", _header("..")) == "data"
    assert _name("https://portal.example/", {}) == "data"


def test_name_uses_the_header_or_the_url():
    assert _name(URL, _header("export.xlsx")) == "export.xlsx"
    assert _name(URL, {}) == "table.csv"
