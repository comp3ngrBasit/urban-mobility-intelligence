"""Shared supervised-learning loop used by the congestion, speed and travel-time models.

1. Chronological split  train | valid | test  (by timestamp, never shuffled)
2. Per algorithm: Optuna search fit-on-train / score-on-valid
3. TimeSeriesSplit CV on the training period for the tuned config (stability estimate)
4. Refit tuned config on train+valid, score once on the untouched test period
5. Compare against naive baselines evaluated on the same test rows
6. Best model -> SHAP, importance, error analysis, saved + logged to MLflow
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.common.config import model_config
from src.common.logging import get_logger
from src.evaluation.explain import feature_importance, shap_summary
from src.evaluation.metrics import classification_metrics, error_breakdown, regression_metrics
from src.models.algorithms import make_model, search_space
from src.models.registry import MlflowRun, new_version_dir, save_version

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def chronological_split(df: pd.DataFrame, ts_col: str, train_frac: float, valid_frac: float):
    ts = np.sort(df[ts_col].unique())
    t1 = ts[int(len(ts) * train_frac)]
    t2 = ts[int(len(ts) * (train_frac + valid_frac))]
    train = df[df[ts_col] < t1]
    valid = df[(df[ts_col] >= t1) & (df[ts_col] < t2)]
    test = df[df[ts_col] >= t2]
    return train, valid, test, (pd.Timestamp(t1), pd.Timestamp(t2))


@dataclass
class TaskSpec:
    task: str                         # e.g. "speed_h60"
    kind: str                         # "reg" | "clf"
    target: str
    features: list[str]
    ts_col: str
    algorithms: list[str]
    labels: list[str] | None = None   # classification label order
    baselines: dict[str, Callable[[pd.DataFrame], np.ndarray]] = field(default_factory=dict)
    error_groups: list[str] = field(default_factory=list)
    extra_config: dict[str, Any] = field(default_factory=dict)
    max_train_rows: int = 2_000_000
    tune_rows: int = 300_000              # Optuna + CV run on the most recent rows to bound cost
    target_transform: str | None = None   # "log1p" for count targets


def _xy(df: pd.DataFrame, spec: TaskSpec):
    X = df[spec.features].apply(pd.to_numeric, errors="coerce").astype("float32")
    y = df[spec.target]
    if spec.kind == "clf":
        y = pd.Categorical(y.astype(str), categories=spec.labels).codes
    elif spec.target_transform == "log1p":
        y = np.log1p(y.astype(float))
    return X, np.asarray(y)


def _inverse(spec: TaskSpec, yhat):
    return np.expm1(yhat).clip(min=0) if spec.target_transform == "log1p" else yhat


def _score(spec: TaskSpec, model, X, y) -> float:
    """Higher is better: -RMSE for regression, macro-F1 for classification."""
    pred = model.predict(X)
    if spec.kind == "reg":
        return -float(np.sqrt(np.mean((np.asarray(pred, float) - y) ** 2)))
    from sklearn.metrics import f1_score
    return float(f1_score(y, np.asarray(pred).ravel().astype(int), average="macro"))


def _fit(name: str, spec: TaskSpec, params: dict, X, y):
    m = make_model(name, spec.kind, params, n_classes=len(spec.labels or []))
    m.fit(X, y)
    return m


def _eval(spec: TaskSpec, model, X, y_raw) -> tuple[dict, np.ndarray, np.ndarray | None]:
    if spec.kind == "reg":
        pred = _inverse(spec, model.predict(X))
        return regression_metrics(y_raw, pred), pred, None
    proba = model.predict_proba(X)
    codes = np.argmax(proba, axis=1)
    pred = np.array(spec.labels)[codes]
    return classification_metrics(np.asarray(y_raw).astype(str), pred, proba, spec.labels), pred, proba


def train_task(df: pd.DataFrame, spec: TaskSpec, metadata: dict | None = None) -> dict:
    cfg = model_config()
    df = df.dropna(subset=[spec.target]).sort_values(spec.ts_col)
    train, valid, test, (t1, t2) = chronological_split(df, spec.ts_col, cfg["split"]["train_frac"],
                                                       cfg["split"]["valid_frac"])
    if len(train) > spec.max_train_rows:   # most recent rows are most relevant
        train = train.iloc[-spec.max_train_rows:]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"{spec.task}: not enough data for a three-way chronological split")
    # drop features with no observations in the most recent training window (e.g. 7-day
    # lags on a short archive) — they carry no signal and break histogram binning
    recent = train.iloc[-spec.tune_rows:]
    dropped = [f for f in spec.features if recent[f].isna().all() or train[f].isna().all()]
    if dropped:
        log.warning("%s: dropping all-empty features %s", spec.task, dropped)
        spec.features = [f for f in spec.features if f not in dropped]
    Xtr, ytr = _xy(train, spec)
    Xva, yva = _xy(valid, spec)
    Xte, _ = _xy(test, spec)
    y_test_raw = test[spec.target].to_numpy()
    log.info("%s: train=%d valid=%d test=%d (cut %s / %s)", spec.task, len(train), len(valid),
             len(test), t1, t2)

    results: dict[str, Any] = {"split": {"train_end": str(t1), "valid_end": str(t2),
                                         "train_rows": len(train), "valid_rows": len(valid),
                                         "test_rows": len(test),
                                         "test_period": [str(test[spec.ts_col].min()),
                                                         str(test[spec.ts_col].max())]},
                               "models": {}, "baselines": {}}
    fitted: dict[str, Any] = {}
    with MlflowRun("urban-mobility", spec.task) as mlrun:
        mlrun.log_params({"task": spec.task, "kind": spec.kind, "n_features": len(spec.features),
                          "train_end": t1, "valid_end": t2})
        for name in spec.algorithms:
            n_trials = cfg["optuna_trials"] if name != "random_forest" else max(cfg["optuna_trials"] // 3, 0)

            Xt_, yt_ = Xtr.iloc[-spec.tune_rows:], ytr[-spec.tune_rows:]
            Xv_, yv_ = Xva.iloc[:spec.tune_rows], yva[:spec.tune_rows]

            def objective(trial: optuna.Trial) -> float:
                return _score(spec, _fit(name, spec, search_space(name, trial), Xt_, yt_), Xv_, yv_)

            best_params: dict = {}
            if n_trials > 0:
                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                best_params = study.best_params
            val_model = _fit(name, spec, best_params, Xtr, ytr)
            val_score = _score(spec, val_model, Xva, yva)

            # time-series CV on the training period (stability of the tuned config)
            cv_scores = []
            for tr_idx, te_idx in TimeSeriesSplit(n_splits=cfg["cv_folds"]).split(Xt_):
                m = _fit(name, spec, best_params, Xt_.iloc[tr_idx], yt_[tr_idx])
                cv_scores.append(_score(spec, m, Xt_.iloc[te_idx], yt_[te_idx]))

            final = _fit(name, spec, best_params, pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
            test_metrics, pred, proba = _eval(spec, final, Xte, y_test_raw)
            results["models"][name] = {"params": best_params, "valid_score": val_score,
                                       "cv_scores": cv_scores, "cv_mean": float(np.mean(cv_scores)),
                                       "cv_std": float(np.std(cv_scores)), "test": test_metrics}
            fitted[name] = (final, pred, proba)
            mlrun.log_metrics({"valid_score": val_score, "cv_mean": float(np.mean(cv_scores)),
                               **{k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}},
                              prefix=f"{name}.")
            log.info("%s/%s valid=%.4f cv=%.4f±%.4f", spec.task, name, val_score,
                     np.mean(cv_scores), np.std(cv_scores))

        for bname, fn in spec.baselines.items():
            bpred = fn(test)
            mask = ~pd.isna(bpred)
            if spec.kind == "reg":
                results["baselines"][bname] = regression_metrics(y_test_raw[mask], np.asarray(bpred)[mask])
            else:
                results["baselines"][bname] = classification_metrics(
                    y_test_raw[mask].astype(str), np.asarray(bpred)[mask].astype(str), None, spec.labels)
            results["baselines"][bname]["coverage"] = float(mask.mean())
            mlrun.log_metrics({k: v for k, v in results["baselines"][bname].items()
                               if isinstance(v, (int, float))}, prefix=f"baseline.{bname}.")

        # model selection on VALIDATION score (test is reported, never used to choose)
        best = max(results["models"], key=lambda k: results["models"][k]["valid_score"])
        results["best_model"] = best
        model, pred, proba = fitted[best]
        out = test[[spec.ts_col] + [c for c in spec.error_groups if c in test.columns]].copy()
        out["actual"] = y_test_raw
        out["predicted"] = pred
        if proba is not None:
            for i, lab in enumerate(spec.labels):
                out[f"p_{lab}"] = proba[:, i]
        extras = {"importance": feature_importance(model, spec.features),
                  "shap": shap_summary(model, Xte, cfg["shap_sample"],
                                       class_index=(spec.labels.index(spec.labels[0])
                                                    if spec.kind == "clf" else None))}
        if spec.kind == "reg":
            extras["errors"] = error_breakdown(out, "actual", "predicted", spec.error_groups)
        d = new_version_dir(spec.task)
        save_version(d, model=model, metrics=results,
                     feature_config={"features": spec.features, "target": spec.target,
                                     "kind": spec.kind, "labels": spec.labels,
                                     "target_transform": spec.target_transform, **spec.extra_config},
                     metadata={**(metadata or {}), "task": spec.task, "best_model": best,
                               "rows_total": int(len(df))},
                     extras=extras, frames={"test_predictions": out})
        mlrun.log_artifacts(d)
    return results
