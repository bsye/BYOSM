from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from .config import find_workspace
from .db import Experiment, Report, Result, Specialist, get_or_add, session

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .spec import TaskSpec

WEIGHTS = "model.pt"
EXPERIMENT_FIELDS = ("spec", "meta", "proposal", "errors", "created")


class Project:
    def __init__(self, name: str, ws: Path | None = None):
        self.ws = ws or find_workspace()
        self.name = name
        self.root = self.ws / "specialists" / name

    def _id(self, s: Session) -> int | None:
        return s.scalar(select(Specialist.id).where(Specialist.name == self.name))

    def _need(self, s: Session) -> int:
        sid = self._id(s)
        if sid is None:
            raise KeyError(f"no specialist {self.name!r}")
        return sid

    def exists(self) -> bool:
        with session(self.ws) as s:
            return self._id(s) is not None

    def spec(self) -> TaskSpec:
        from .spec import TaskSpec

        with session(self.ws) as s:
            row = s.execute(select(Specialist.spec, Specialist.created).where(Specialist.name == self.name)).first()
        if row is None:
            raise KeyError(f"no specialist {self.name!r}")
        spec, created = row
        return TaskSpec.model_validate({**spec, "created": spec.get("created") or created.replace(tzinfo=UTC)})

    def save_spec(self, spec: TaskSpec) -> None:
        data = spec.model_dump(mode="json")
        with session(self.ws) as s:
            row = s.scalar(select(Specialist).where(Specialist.name == self.name))
            if row is None:
                s.add(Specialist(name=self.name, spec=data))
            else:
                row.spec = data

    def add_baselines(self, ids: list[str]) -> list[str]:
        with session(self.ws) as s:
            row = s.scalar(select(Specialist).where(Specialist.name == self.name))
            if row is None:
                raise KeyError(f"no specialist {self.name!r}")
            have = list(row.spec.get("baselines", []))
            row.spec = {**row.spec, "baselines": [*have, *(i for i in ids if i not in have)]}
            return row.spec["baselines"]

    def delete(self) -> None:
        with session(self.ws) as s:
            s.execute(delete(Specialist).where(Specialist.name == self.name))

    @property
    def promoted(self) -> str | None:
        with session(self.ws) as s:
            return s.scalar(select(Specialist.promoted).where(Specialist.id == self._need(s)))

    def promote(self, exp: str) -> None:
        with session(self.ws) as s:
            s.scalar(select(Specialist).where(Specialist.name == self.name)).promoted = exp

    def data_dir_for(self, spec: TaskSpec) -> Path:
        return self.root / "data" / spec.data_key()

    @property
    def data_dir(self) -> Path:
        return self.data_dir_for(self.exp_spec(self.promoted) if self.promoted else self.spec())

    @property
    def reviewed_path(self) -> Path:
        return self.root / "reviewed.jsonl"

    @property
    def runs_dir(self) -> Path:
        return self.root / "experiments"

    def exp_dir(self, exp: str) -> Path:
        return self.runs_dir / exp

    @property
    def best_dir(self) -> Path:
        return self.exp_dir(self.promoted) if self.promoted else self.runs_dir / "__none__"

    @property
    def evals_dir(self) -> Path:
        return self.root / "evals"

    def _experiment(self, s: Session, exp: str) -> Experiment | None:
        return s.scalar(select(Experiment).where(Experiment.specialist_id == self._need(s), Experiment.name == exp))

    def experiments(self) -> list[str]:
        with session(self.ws) as s:
            sid = self._id(s)
            mine = Experiment.specialist_id == sid
            return sorted(s.scalars(select(Experiment.name).where(mine, Experiment.meta.is_not(None)))) if sid else []

    def attempts(self) -> list[str]:
        with session(self.ws) as s:
            sid = self._id(s)
            return sorted(s.scalars(select(Experiment.name).where(Experiment.specialist_id == sid))) if sid else []

    def experiment(self, exp: str) -> dict[str, Any] | None:
        with session(self.ws) as s:
            e = self._experiment(s, exp)
            return {k: getattr(e, k) for k in EXPERIMENT_FIELDS} if e else None

    def exp_spec(self, exp: str) -> TaskSpec:
        from .spec import TaskSpec

        e = self.experiment(exp)
        return TaskSpec.model_validate(e["spec"]) if e else self.spec()

    def meta(self, exp: str) -> dict:
        e = self.experiment(exp)
        return (e or {}).get("meta") or {}

    def proposal(self, exp: str) -> dict | None:
        return (self.experiment(exp) or {}).get("proposal")

    def save_experiment(self, exp: str, **fields: Any) -> None:
        if "spec" in fields and not isinstance(fields["spec"], dict):
            fields["spec"] = fields["spec"].model_dump(mode="json")
        with session(self.ws) as s:
            base = s.scalar(select(Specialist.spec).where(Specialist.name == self.name))
            e = get_or_add(s, Experiment, specialist_id=self._need(s), name=exp)
            e.spec = e.spec or base
            for k, v in fields.items():
                setattr(e, k, v)

    def board(self) -> dict[str, Any]:
        if not self.exists():
            return {}
        spec = self.spec()
        keep_ids = {*self.experiments(), *spec.baselines}
        prim = spec.primary_suite()
        order = [prim, *(e.name for e in spec.evals if e.name != prim)]
        with session(self.ws) as s:
            rows = s.scalars(select(Result).where(Result.specialist_id == self._need(s))).all()
        kept = sorted((r for r in rows if r.predictor in keep_ids and r.suite in order), key=lambda r: r.id)
        out: dict[str, dict[str, Any]] = {}
        for r in sorted(kept, key=lambda r: order.index(r.suite)):
            out.setdefault(r.predictor, {})[r.suite] = r.data
        return out

    def result(self, pid: str, suite: str) -> dict | None:
        with session(self.ws) as s:
            return s.scalar(select(Result.data).filter_by(specialist_id=self._need(s), predictor=pid, suite=suite))

    def save_result(self, pid: str, suite: str, data: dict) -> None:
        with session(self.ws) as s:
            r = get_or_add(s, Result, specialist_id=self._need(s), predictor=pid, suite=suite)
            r.kind, r.objective, r.fingerprint, r.data = (
                data["kind"],
                data.get("objective"),
                data.get("fingerprint"),
                data,
            )

    def drop_result(self, pid: str, suite: str) -> None:
        with session(self.ws) as s:
            s.execute(delete(Result).filter_by(specialist_id=self._need(s), predictor=pid, suite=suite))

    def report(self, kind: str) -> Any:
        with session(self.ws) as s:
            return s.scalar(select(Report.data).filter_by(specialist_id=self._id(s), kind=kind))

    def reports(self, prefix: str) -> dict[str, Any]:
        with session(self.ws) as s:
            found = select(Report.kind, Report.data).filter_by(specialist_id=self._id(s))
            return dict(s.execute(found.where(Report.kind.startswith(prefix))).all())

    def save_report(self, kind: str, data: Any) -> None:
        with session(self.ws) as s:
            get_or_add(s, Report, specialist_id=self._need(s), kind=kind).data = data

    def ensure(self) -> Project:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        return self


def list_projects(ws: Path | None = None) -> list[str]:
    with session(ws or find_workspace()) as s:
        return sorted(s.scalars(select(Specialist.name)))


def free_name(base: str, ws: Path | None = None) -> str:
    taken = set(list_projects(ws))
    return next(n for n in (base, *(f"{base}-{i}" for i in range(2, len(taken) + 3))) if n not in taken)
