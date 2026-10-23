from __future__ import annotations

from typing import Any

import numpy as np

from .snapshots import CUTOFF, LABEL

BOOTSTRAP = 200
SEED = 0


def _xy(df: Any, categorical: list[str]) -> tuple[Any, Any]:
    x = df.drop(columns=[LABEL, CUTOFF]).copy()
    for c in categorical:
        x[c] = x[c].astype("category")
    return x, df[LABEL].to_numpy()


def _ci(y: Any, p: Any, metric: Any) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(metric(y[i], p[i]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def scores(y: Any, p: Any) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auc": float(roc_auc_score(y, p)),
        "auc_ci": _ci(y, p, roc_auc_score),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
    }


def models(categorical: list[str]) -> dict[str, Any]:
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    def numeric(x: Any) -> list[str]:
        return [c for c in x.columns if c not in categorical]

    encode = make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=50)
            ),
            categorical,
        ),
        (SimpleImputer(), numeric),
    )
    return {
        "gradient boosting": HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED),
        "logistic regression": make_pipeline(encode, LogisticRegression(max_iter=1000)),
    }


def evaluate(splits: dict[str, Any], categorical: list[str]) -> dict[str, Any]:
    xtr, ytr = _xy(splits["train"], categorical)
    xte, yte = _xy(splits["test"], categorical)
    out = {"base rate": scores(yte, np.full(len(yte), ytr.mean()))}
    for name, model in models(categorical).items():
        cols = xtr if name == "gradient boosting" else xtr.astype({c: object for c in categorical})
        test = xte if name == "gradient boosting" else xte.astype({c: object for c in categorical})
        model.fit(cols, ytr)
        out[name] = scores(yte, model.predict_proba(test)[:, 1])
    return out
