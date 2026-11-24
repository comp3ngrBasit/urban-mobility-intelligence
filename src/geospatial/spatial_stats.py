"""Spatial autocorrelation: Global Moran's I, Local Moran's I (LISA) and Getis-Ord Gi*
on H3-aggregated counts, with permutation inference and Benjamini-Hochberg FDR control."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.geospatial.h3utils import neighbors


def h3_weights(cells: list[str]):
    """Queen-like contiguity on the H3 grid (k=1 ring), row-standardised."""
    from libpysal.weights import W

    idx = set(cells)
    nb = {c: [n for n in neighbors(c, 1) if n in idx] for c in cells}
    # islands get no neighbours; libpysal handles them (warns), stats exclude them implicitly
    w = W(nb, silence_warnings=True)
    w.transform = "r"
    return w


def global_morans_i(values: pd.Series, permutations: int = 999, seed: int = 42) -> dict:
    from esda.moran import Moran

    w = h3_weights(list(values.index))
    np.random.seed(seed)  # esda.Moran has no seed argument
    m = Moran(values.loc[w.id_order].to_numpy(dtype=float), w, permutations=permutations)
    return {"I": float(m.I), "expected_I": float(m.EI), "z": float(m.z_sim),
            "p_value": float(m.p_sim), "n": int(m.n), "permutations": permutations}


def local_clusters(values: pd.Series, alpha: float = 0.05, permutations: int = 999,
                   seed: int = 42) -> pd.DataFrame:
    """Return per-cell LISA quadrant + Gi* z-score, both FDR-adjusted."""
    from esda.getisord import G_Local
    from esda.moran import Moran_Local

    w = h3_weights(list(values.index))
    y = values.loc[w.id_order].to_numpy(dtype=float)
    lisa = Moran_Local(y, w, permutations=permutations, seed=seed)
    gi = G_Local(y, w, star=True, permutations=permutations, seed=seed, transform="B")
    lisa_sig = multipletests(lisa.p_sim, alpha=alpha, method="fdr_bh")[0]
    gi_sig = multipletests(gi.p_sim, alpha=alpha, method="fdr_bh")[0]
    quad = np.array(["", "HH", "LH", "LL", "HL"])[lisa.q]
    df = pd.DataFrame({
        "h3": w.id_order, "value": y, "local_i": lisa.Is, "lisa_p": lisa.p_sim,
        "lisa_cluster": np.where(lisa_sig, quad, "ns"),
        "gi_z": gi.Zs, "gi_p": gi.p_sim,
        "gi_class": np.where(gi_sig & (gi.Zs > 0), "Hotspot",
                             np.where(gi_sig & (gi.Zs < 0), "Coldspot", "Not significant")),
    })
    return df
