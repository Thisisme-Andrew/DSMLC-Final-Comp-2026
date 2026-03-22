# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 5 — Apply model to other wind farms
# MAGIC
# MAGIC - **Farm A → B:** Option **A** = feature intersection (shared column names); Option **B** = shared latent (sklearn baseline: per-farm scaler + PCA).
# MAGIC - **Farm C:** Fresh **Isolation Forest** (+ real **LSTM** in `03b`); optional **SimCLR** notes for semi-supervised pretrain.
# MAGIC - **Eval:** `care_style_metrics` aligns with point-level CARE-style checks from Stage 4.
# MAGIC
# MAGIC **Reminder:** Farm A may only show statuses `0, 3, 4` — do not assume status `2` exists on B/C.

# COMMAND ----------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 1) Status distributions ---


def status_distribution(df: pd.DataFrame, col: str = "status_type_id") -> pd.Series:
    if col not in df.columns:
        alt = "status_type" if col == "status_type_id" else "status_type_id"
        if alt in df.columns:
            col = alt
        else:
            raise KeyError(f"No status column; expected {col} or {alt}")
    return df[col].value_counts(dropna=False).sort_index()


def compare_status_across_farms(
    farm_frames: dict[str, pd.DataFrame], col: str = "status_type_id"
) -> pd.DataFrame:
    series_list = []
    for name, d in farm_frames.items():
        vc = status_distribution(d, col)
        vc.name = name
        series_list.append(vc)
    return pd.concat(series_list, axis=1).fillna(0).astype(int)


# --- 2) Option A: common sensors A ∩ B ---

DEFAULT_NON_FEATURE_PREFIXES: tuple[str, ...] = (
    "asset_id",
    "time_stamp",
    "timestamp",
    "id",
    "train_test",
    "status_type",
    "event",
    "farm",
    "label",
    "y_",
)


def _is_candidate_feature(name: str) -> bool:
    n = name.lower()
    if any(n.startswith(p) or n == p.rstrip("_") for p in DEFAULT_NON_FEATURE_PREFIXES):
        return False
    return True


def sensor_like_columns(cols: Iterable[str]) -> list[str]:
    return [c for c in cols if _is_candidate_feature(c)]


def common_sensor_columns(
    cols_a: Sequence[str],
    cols_b: Sequence[str],
    restrict_to_sensor_like: bool = True,
) -> list[str]:
    sa = set(sensor_like_columns(cols_a) if restrict_to_sensor_like else cols_a)
    sb = set(sensor_like_columns(cols_b) if restrict_to_sensor_like else cols_b)
    return sorted(sa & sb)


def intersection_option_a(
    df: pd.DataFrame, common_cols: Sequence[str], fillna: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in common_cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing intersection columns: {missing[:10]}...")
    X = df.loc[:, list(common_cols)].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(fillna)
    return X, list(common_cols)

# COMMAND ----------

# --- 3) Option B: shared latent (sklearn baseline; swap for PyTorch dual encoders if needed) ---


@dataclass
class SharedLatentOptionB:
    latent_dim: int = 32
    random_state: int = 42
    scaler_a: StandardScaler | None = field(default=None, init=False)
    scaler_b: StandardScaler | None = field(default=None, init=False)
    pca_: PCA | None = field(default=None, init=False)

    def fit(self, X_a: np.ndarray, X_b: np.ndarray) -> SharedLatentOptionB:
        if X_a.shape[1] != X_b.shape[1]:
            raise ValueError("A and B must use the same intersection feature dimension")
        self.scaler_a = StandardScaler().fit(X_a)
        self.scaler_b = StandardScaler().fit(X_b)
        Za = self.scaler_a.transform(X_a)
        Zb = self.scaler_b.transform(X_b)
        stacked = np.vstack([Za, Zb])
        n_comp = min(self.latent_dim, stacked.shape[1], stacked.shape[0])
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state).fit(stacked)
        return self

    def transform_a(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler_a is not None and self.pca_ is not None
        return self.pca_.transform(self.scaler_a.transform(X))

    def transform_b(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler_b is not None and self.pca_ is not None
        return self.pca_.transform(self.scaler_b.transform(X))


# --- 4) Farm C: fresh IF; MLP autoencoder skeleton (use 03b for real LSTM) ---


def farm_c_isolation_forest_baseline(
    X_train: np.ndarray, contamination: float = 0.05, random_state: int = 42
) -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    ).fit(X_train)


def sklearn_lstm_style_autoencoder_skeleton(
    n_features: int, latent_dim: int = 32, random_state: int = 42
) -> Pipeline:
    hidden = max(latent_dim * 2, 64)
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden, latent_dim, hidden),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("ae", mlp)])


def simclr_pretrain_stub_notebook_cells() -> str:
    return """
# SimCLR-style (Farm C, advanced) — GPU notebook

1. Window multivariate series; augment (jitter, scale, time-shift).
2. Encoder: 1D CNN or small Transformer -> projection head -> 128-d.
3. NT-Xent on two views of the same window.
4. Fine-tune with rare labels (e.g. 12 events).
"""

# COMMAND ----------

# --- 5) CARE-style metrics + Stage 5 sanity report ---


def care_style_metrics(
    y_true_anomaly: np.ndarray,
    alert_binary: np.ndarray,
    normal_mask: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true_anomaly).astype(int)
    a = np.asarray(alert_binary).astype(int)
    tp = int(np.sum((y == 1) & (a == 1)))
    fp = int(np.sum((y == 0) & (a == 1)))
    fn = int(np.sum((y == 1) & (a == 0)))
    tn = int(np.sum((y == 0) & (a == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out: dict[str, float] = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    if normal_mask is not None:
        nm = np.asarray(normal_mask).astype(bool)
        if nm.any():
            fa_rate = float(np.mean(a[nm]))
            out["reliability_proxy"] = float(1.0 - fa_rate)
    return out


def run_stage5_report(
    df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame | None = None
) -> dict[str, Any]:
    farms = {"farm_a": df_a, "farm_b": df_b}
    if df_c is not None:
        farms["farm_c"] = df_c
    status_tbl = compare_status_across_farms(farms)
    common_ab = common_sensor_columns(df_a.columns, df_b.columns)
    report = {
        "n_features_a": len(sensor_like_columns(df_a.columns)),
        "n_features_b": len(sensor_like_columns(df_b.columns)),
        "n_common_a_b": len(common_ab),
        "sample_common_columns": common_ab[:20],
        "status_counts_by_farm": status_tbl.to_dict(),
        "simclr_notebook_hint": simclr_pretrain_stub_notebook_cells().strip()[:200] + "...",
    }
    if df_c is not None:
        report["n_features_c"] = len(sensor_like_columns(df_c.columns))
    return report

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load farms from Unity Catalog (optional)
# MAGIC
# MAGIC Adjust `CATALOG` / `SCHEMA` and table names. Use sampling while iterating to save memory.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "wind-turbine-silver"


def fq(name: str) -> str:
    return f"`{CATALOG}`.`{SCHEMA}`.`{name}`"


# Example (uncomment and tune limits):
# df_a = spark.table(fq("wind-farm-a-features")).limit(50_000).toPandas()
# df_b = spark.table(fq("wind-farm-b-features")).limit(50_000).toPandas()
# df_c = spark.table(fq("wind-farm-c-features")).limit(50_000).toPandas()
# report = run_stage5_report(df_a, df_b, df_c)
# print(json.dumps({k: v for k, v in report.items() if k != "status_counts_by_farm"}, indent=2))
# display(compare_status_across_farms({"A": df_a, "B": df_b}))

print("Stage 5 notebook ready — load df_a/df_b/df_c then call run_stage5_report.")