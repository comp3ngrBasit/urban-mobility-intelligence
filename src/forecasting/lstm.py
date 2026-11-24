"""Optional global LSTM for zone-hour demand (one model, all zones, zone embedding).
Only imported when configs/models.yaml demand.enable_lstm is true and torch is installed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.common.config import get_settings, model_config
from src.evaluation.metrics import regression_metrics
from src.features.demand import complete_panel
from src.models.registry import new_version_dir, save_version

WINDOW = 168


class DemandLSTM(nn.Module):
    def __init__(self, n_zones: int, horizons: int, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_zones, 8)
        self.lstm = nn.LSTM(1 + 8, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.head = nn.Linear(hidden, horizons)

    def forward(self, seq, zone):
        e = self.emb(zone).unsqueeze(1).expand(-1, seq.shape[1], -1)
        out, _ = self.lstm(torch.cat([seq, e], dim=-1))
        return self.head(out[:, -1])


def _windows(mat: np.ndarray, starts, horizons):
    X, Y, Z = [], [], []
    for z in range(mat.shape[0]):
        for s in starts:
            X.append(mat[z, s - WINDOW:s])
            Y.append([mat[z, s + h - 1] for h in horizons])
            Z.append(z)
    return (torch.tensor(np.array(X)[..., None], dtype=torch.float32),
            torch.tensor(np.array(Y), dtype=torch.float32), torch.tensor(Z))


def train_lstm(epochs: int = 8) -> dict:
    mc = model_config()
    horizons = mc["demand"]["horizons_hours"]
    panel = complete_panel(pd.read_parquet(get_settings().processed_dir / "tlc_pickups_zone_hour.parquet"))
    wide = panel.pivot(index="zone", columns="ts", values="pickups").fillna(0)
    mat = np.log1p(wide.to_numpy(float))
    T = mat.shape[1]
    cut = int(T * (mc["split"]["train_frac"] + mc["split"]["valid_frac"]))
    tr_starts = range(WINDOW, cut - max(horizons), 6)
    te_starts = range(cut, T - max(horizons) + 1, 24)
    Xtr, Ytr, Ztr = _windows(mat, tr_starts, horizons)
    Xte, Yte, Zte = _windows(mat, te_starts, horizons)
    model = DemandLSTM(mat.shape[0], len(horizons))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    torch.manual_seed(42)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b], Ztr[b]), Ytr[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        P = model(Xte, Zte).numpy()
    res = {f"lstm_h{h}": regression_metrics(np.expm1(Yte[:, i].numpy()), np.expm1(P[:, i]).clip(0))
           for i, h in enumerate(horizons)}
    d = new_version_dir("demand_lstm")
    torch.save(model.state_dict(), d / "lstm.pt")
    save_version(d, metrics=res, feature_config={"window_h": WINDOW, "horizons": horizons},
                 metadata={"epochs": epochs})
    return res
