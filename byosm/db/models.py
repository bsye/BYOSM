from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB(), "postgresql")


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: Json, list[Any]: Json, Any: Json}


class Specialist(Base):
    __tablename__ = "specialists"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    spec: Mapped[dict[str, Any]]
    promoted: Mapped[str | None] = mapped_column(String(200))
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


def _owned() -> Mapped[int]:
    return mapped_column(ForeignKey("specialists.id", ondelete="CASCADE"), index=True)


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (UniqueConstraint("specialist_id", "name"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    specialist_id: Mapped[int] = _owned()
    name: Mapped[str] = mapped_column(String(200))
    spec: Mapped[dict[str, Any]]
    meta: Mapped[dict[str, Any] | None]
    proposal: Mapped[dict[str, Any] | None]
    errors: Mapped[list[Any] | None]
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Result(Base):
    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("specialist_id", "predictor", "suite"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    specialist_id: Mapped[int] = _owned()
    predictor: Mapped[str] = mapped_column(String(300))
    kind: Mapped[str] = mapped_column(String(20))
    suite: Mapped[str] = mapped_column(String(200))
    objective: Mapped[float | None] = mapped_column(Float)
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    data: Mapped[dict[str, Any]]
    scored: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("specialist_id", "kind"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    specialist_id: Mapped[int] = _owned()
    kind: Mapped[str] = mapped_column(String(100))
    data: Mapped[Any] = mapped_column(Json)
    updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Row(Base):
    __tablename__ = "rows"
    __table_args__ = (Index("rows_lookup", "specialist_id", "collection", "split", "position"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    specialist_id: Mapped[int | None] = mapped_column(ForeignKey("specialists.id", ondelete="CASCADE"))
    collection: Mapped[str] = mapped_column(String(200))
    split: Mapped[str] = mapped_column(String(50))
    position: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict[str, Any]]


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    args: Mapped[list[Any]]
    status: Mapped[str] = mapped_column(String(20))
    created: Mapped[float | None] = mapped_column(Float)
    started: Mapped[float | None] = mapped_column(Float)
    ended: Mapped[float | None] = mapped_column(Float)
    code: Mapped[int | None] = mapped_column(Integer)
    pid: Mapped[int | None] = mapped_column(Integer)


class JobLine(Base):
    __tablename__ = "job_lines"
    id: Mapped[int] = mapped_column(primary_key=True)
    job: Mapped[str] = mapped_column(String(40), index=True)
    line: Mapped[str] = mapped_column(Text)


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    job: Mapped[str] = mapped_column(String(40), index=True)
    kind: Mapped[str | None] = mapped_column(String(40), index=True)
    model: Mapped[str | None] = mapped_column(String(200))
    data: Mapped[dict[str, Any]]
