import os

import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from byosm.data.store import Dataset
from byosm.db.engine import URL_ENV, database_url, engine
from byosm.db.json_field import JsonField
from byosm.db.models import ROW_LABEL, ROWS_BY_LABEL, Row

ROWS = [
    {"input": "a", "label": "yes"},
    {"input": "b", "label": "no"},
    {"input": "c", "label": ["B-LOC", "O", "O"]},
]
sqlite_only = pytest.mark.skipif(bool(os.environ.get(URL_ENV)), reason="reads the SQLite query plan")


def test_label_counts_include_every_tag_of_a_tagged_row(ws):
    d = Dataset(ws / "data" / "k")
    d.save("train", ROWS)
    assert d.summary()["train"] == {"n": 3, "labels": {"yes": 1, "no": 1, "B-LOC": 1, "O": 2}}


@sqlite_only
def test_label_counts_are_read_from_the_label_index(ws):
    d = Dataset(ws / "data" / "k")
    d.save("train", ROWS)
    eng, sent = engine(database_url(ws)), []

    def record(conn, cursor, statement, params, context, many):
        sent.append((statement, params))

    event.listen(eng, "before_cursor_execute", record)
    try:
        d.summary()
    finally:
        event.remove(eng, "before_cursor_execute", record)
    counts = [(q, p) for q, p in sent if "GROUP BY" in q]
    with eng.connect() as c:
        plans = [[r[-1] for r in c.exec_driver_sql("EXPLAIN QUERY PLAN " + q, p).all()] for q, p in counts]
    assert len(plans) == 4
    assert all(any(ROWS_BY_LABEL.name in step for step in plan) for plan in plans)
    assert not any("TEMP B-TREE" in step for plan in plans for step in plan)


@sqlite_only
def test_an_existing_database_gains_a_declared_index_when_the_app_starts(ws):
    url = database_url(ws)
    with engine(url).begin() as c:
        c.exec_driver_sql(f"DROP INDEX {ROWS_BY_LABEL.name}")
    engine.cache_clear()
    with engine(url).connect() as c:
        names = c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'index'").scalars()
        assert ROWS_BY_LABEL.name in set(names)


def test_postgres_reads_the_label_the_way_it_indexes_it():
    dialect = postgresql.dialect()
    assert "(data -> 'label')" in str(CreateIndex(ROWS_BY_LABEL).compile(dialect=dialect))
    assert "(rows.data -> 'label')" in str(select(JsonField(Row.data, ROW_LABEL)).compile(dialect=dialect))


def test_a_field_name_that_is_not_an_identifier_is_refused():
    with pytest.raises(ValueError, match="not a JSON field name"):
        JsonField(Row.data, "label') or ('1")
