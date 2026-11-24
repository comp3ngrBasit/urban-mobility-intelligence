"""Reusable statistical tests that return JSON-ready dicts with effect sizes and CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, conf: float = 0.95) -> dict:
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else None, "lo": None, "hi": None, "n": n}
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "n": int(n)}


def correlation(x, y) -> dict:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 10 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return {"n": len(d), "pearson_r": None, "spearman_rho": None}
    pr, pp = stats.pearsonr(d["x"], d["y"])
    sr, sp = stats.spearmanr(d["x"], d["y"])
    # Fisher-z CI for Pearson r
    z, se = np.arctanh(pr), 1 / np.sqrt(len(d) - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return {"n": int(len(d)), "pearson_r": float(pr), "pearson_p": float(pp),
            "pearson_ci95": [float(lo), float(hi)], "spearman_rho": float(sr), "spearman_p": float(sp)}


def anova(groups: dict[str, pd.Series]) -> dict:
    g = {k: np.asarray(v.dropna(), float) for k, v in groups.items() if v.notna().sum() >= 5}
    if len(g) < 2:
        return {"note": "fewer than two groups with data"}
    f, p = stats.f_oneway(*g.values())
    h, kp = stats.kruskal(*g.values())
    allv = np.concatenate(list(g.values()))
    ss_between = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in g.values())
    ss_total = ((allv - allv.mean()) ** 2).sum()
    return {"anova_F": float(f), "anova_p": float(p), "eta_squared": float(ss_between / ss_total),
            "kruskal_H": float(h), "kruskal_p": float(kp),
            "groups": {k: mean_ci(v) for k, v in g.items()}}


def two_group(a: pd.Series, b: pd.Series, labels=("a", "b")) -> dict:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 5 or len(b) < 5:
        return {"note": "insufficient data"}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, up = stats.mannwhitneyu(a, b)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return {"welch_t": float(t), "welch_p": float(p), "mannwhitney_U": float(u), "mannwhitney_p": float(up),
            "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled else None,
            labels[0]: mean_ci(a), labels[1]: mean_ci(b)}


def chi_square(table: pd.DataFrame) -> dict:
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"note": "degenerate table"}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(v)}


def decompose(series: pd.Series, period: int) -> dict:
    from statsmodels.tsa.seasonal import STL

    s = series.asfreq(series.index.inferred_freq or "h").interpolate(limit=6)
    s = s.dropna()
    if len(s) < 2 * period:
        return {"note": "series too short"}
    r = STL(s, period=period, robust=True).fit()
    var_r = np.var(r.resid)
    return {"ts": [str(t) for t in s.index], "observed": s.round(3).tolist(),
            "trend": r.trend.round(3).tolist(), "seasonal": r.seasonal.round(3).tolist(),
            "resid": r.resid.round(3).tolist(), "period": period,
            "strength_trend": float(max(0, 1 - var_r / np.var(r.trend + r.resid))),
            "strength_seasonal": float(max(0, 1 - var_r / np.var(r.seasonal + r.resid)))}
