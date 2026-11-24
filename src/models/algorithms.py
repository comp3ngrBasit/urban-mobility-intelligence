"""Model factory + Optuna search spaces for every algorithm in the brief."""
from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

SEED = 42


def _fill_empty_columns(X):
    """sklearn's histogram binner crashes on a column that is entirely NaN (possible in an
    early CV fold, e.g. 7-day lags). A constant column is never split on, so this is inert."""
    import numpy as np
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        empty = X.columns[X.isna().all()]
        return X.assign(**{c: 0.0 for c in empty}) if len(empty) else X
    X = np.array(X, dtype=float, copy=True)
    X[:, np.isnan(X).all(axis=0)] = 0.0
    return X


class SafeHGBRegressor(HistGradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(_fill_empty_columns(X), y, sample_weight=sample_weight)


class SafeHGBClassifier(HistGradientBoostingClassifier):
    def fit(self, X, y, sample_weight=None):
        return super().fit(_fill_empty_columns(X), y, sample_weight=sample_weight)


def make_model(name: str, kind: str, params: dict[str, Any] | None = None, n_classes: int = 2):
    """kind: 'reg' | 'clf'. All returned estimators accept NaN inputs."""
    p = dict(params or {})
    if name == "random_forest":
        base = dict(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
        base.update(p)
        est = RandomForestRegressor(**base) if kind == "reg" else \
            RandomForestClassifier(class_weight="balanced_subsample", **base)
        return make_pipeline(SimpleImputer(strategy="median"), est)
    if name == "hist_gradient_boosting":
        base = dict(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED)
        base.update(p)
        return SafeHGBRegressor(**base) if kind == "reg" else \
            SafeHGBClassifier(class_weight="balanced", **base)
    if name == "xgboost":
        import xgboost as xgb
        base = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
        base.update(p)
        if kind == "reg":
            return xgb.XGBRegressor(objective="reg:squarederror", **base)
        return xgb.XGBClassifier(objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                                 eval_metric="mlogloss", **base)
    if name == "lightgbm":
        import lightgbm as lgb
        base = dict(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
        base.update(p)
        return lgb.LGBMRegressor(**base) if kind == "reg" else \
            lgb.LGBMClassifier(class_weight="balanced", **base)
    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        base = dict(iterations=800, learning_rate=0.08, depth=8, random_seed=SEED, verbose=False,
                    allow_writing_files=False)
        base.update(p)
        return CatBoostRegressor(**base) if kind == "reg" else \
            CatBoostClassifier(auto_class_weights="Balanced", loss_function="MultiClass", **base)
    raise ValueError(f"unknown model {name}")


def search_space(name: str, trial: optuna.Trial) -> dict[str, Any]:
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
    if name == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True)}
    if name == "xgboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "lightgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True)}
    raise ValueError(name)

