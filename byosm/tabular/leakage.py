from __future__ import annotations

from typing import Any

from .snapshots import CUTOFF, LABEL

SUSPECT_AUC = 0.85
SUSPECT_MISSING_GAP = 0.3
MIN_GROUP = 200
PURE_RATE = 0.02
PURE_POSITIVE = 0.95
SUSPECT_PURE_SHARE = 0.2


def _auc(x: Any, y: Any) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    model = HistGradientBoostingClassifier(max_iter=50, categorical_features="from_dtype")
    p = cross_val_predict(model, x.to_frame(), y, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def pure_share(x: Any, y: Any) -> float:
    groups = y.groupby(x, observed=True).agg(["mean", "size", "sum"])
    big = groups[groups["size"] >= MIN_GROUP]
    base = float(y.mean())
    ended = big[big["mean"] >= PURE_POSITIVE]["sum"].sum() / max(1, y.sum())
    never = big[big["mean"] <= base * PURE_RATE]
    survived = (never["size"] - never["sum"]).sum() / max(1, len(y) - y.sum())
    return float(max(ended, survived))


def check(train: Any, categorical: list[str]) -> list[dict[str, Any]]:
    train = train[train[CUTOFF] == train[CUTOFF].max()]
    y = train[LABEL]
    out = []
    for col in [c for c in train.columns if c not in (LABEL, CUTOFF)]:
        x = train[col].astype("category") if col in categorical else train[col]
        missing = train[col].isna()
        gap = abs(float(missing[y == 1].mean()) - float(missing[y == 0].mean()))
        auc = _auc(x, y)
        pure = pure_share(train[col], y)
        why = []
        if auc >= SUSPECT_AUC:
            why.append(f"alone it predicts the label with AUC {auc:.2f}")
        if gap >= SUSPECT_MISSING_GAP:
            why.append(
                f"missing for {missing[y == 1].mean():.0%} of positives but {missing[y == 0].mean():.0%} of negatives"
            )
        if pure >= SUSPECT_PURE_SHARE:
            why.append(
                f"{pure:.0%} of one class sits in values that are almost all that class, as if written after the outcome"
            )
        out.append(
            {
                "feature": col,
                "auc": round(auc, 3),
                "missing_gap": round(gap, 3),
                "suspect": bool(why),
                "why": "; ".join(why),
            }
        )
    return sorted(out, key=lambda r: -r["auc"])
