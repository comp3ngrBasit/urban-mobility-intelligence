"""Evaluation metrics and error analysis. Nothing here is ever hand-entered: all numbers
come from predictions on held-out, chronologically later data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"mae": float(mean_absolute_error(y_true, y_pred)),
           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "r2": float(r2_score(y_true, y_pred)), "n": int(len(y_true))}
    # MAPE only where the denominator is meaningfully non-zero
    mask = np.abs(y_true) >= 1.0
    out["mape_pct"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
        if mask.sum() > 0.5 * len(y_true) else None
    out["mape_note"] = None if out["mape_pct"] is not None else "undefined: too many near-zero targets"
    return out


def classification_metrics(y_true, y_pred, proba: np.ndarray | None, labels: list[str]) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
           "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
           "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                              zero_division=0),
           "confusion_matrix": {"labels": labels,
                                "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
           "n": int(len(y_true))}
    if proba is not None:
        present = [l for l in labels if l in set(y_true)]
        if len(present) >= 2:
            try:
                idx = [labels.index(l) for l in present]
                p = proba[:, idx] / proba[:, idx].sum(axis=1, keepdims=True)
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, p, multi_class="ovr",
                                                               labels=present, average="macro"))
            except ValueError:
                out["roc_auc_ovr_macro"] = None
    return out


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def error_breakdown(df: pd.DataFrame, y: str, yhat: str, by: list[str], top: int = 50) -> dict:
    d = df.assign(abs_err=(df[y] - df[yhat]).abs(), err=df[yhat] - df[y])
    out = {}
    for col in by:
        if col in d.columns:
            g = d.groupby(col).agg(mae=("abs_err", "mean"), bias=("err", "mean"), n=("err", "size"))
            out[col] = g.sort_values("mae", ascending=False).head(top).reset_index().to_dict("records")
    resid = d["err"].to_numpy()
    out["residual_hist"] = dict(zip(["counts", "edges"],
                                    [a.tolist() for a in np.histogram(resid[np.isfinite(resid)], bins=40)]))
    return out
