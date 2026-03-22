# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 4 — CARE Ensemble
# MAGIC
# MAGIC ## No-leakage rules (this notebook)
# MAGIC - **MinMaxScaler** is fit on `train_test == 'train'` rows only, then applied to all rows.
# MAGIC - **Weights** for the weighted ensemble are computed from **validation windows only**: `train_test == 'train'` rows that fall inside labeled intervals in `wind-farm-a-event-info` (never `prediction`).
# MAGIC - **Threshold selection** for `alert_triggered` compares candidate thresholds on those same validation windows only (grid search maximizing F1 vs. point-in-window labels). **Test / prediction rows are not used for tuning.**
# MAGIC - **Simple equal-weight average** (`ensemble_simple`) is reported first; **weighted** `ensemble_score` uses validation-derived weights. Correlation between the two is printed for an explicit comparison.
# MAGIC
# MAGIC ## Output tables (Unity Catalog)
# MAGIC - `workspace.wind-turbine-silver.wind-farm-a-ensemble-alerts`
# MAGIC - `workspace.wind-turbine-silver.wind-farm-a-alert-summary`
# MAGIC - `workspace.wind-turbine-silver.care-score-summary` (intermediate)
# MAGIC
# MAGIC > **Note:** `wind-farm-a-lstm-scores` exposes reconstruction error as `lstm_recon_error`; it is aliased to **`lstm_anomaly_score`** for the ensemble contract. `status_type` is derived from `status_type_id`.

# COMMAND ----------

# CELL 1 — Imports and config
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Unity Catalog — adjust if your catalog/schema differ
CATALOG = "workspace"
SCHEMA = "wind-turbine-silver"

def fq(name: str) -> str:
    """Fully-qualified Spark table name with backticks for hyphenated identifiers."""
    return f"`{CATALOG}`.`{SCHEMA}`.`{name}`"

def table_exists(name: str) -> bool:
    try:
        spark.sql(f"DESCRIBE TABLE {fq(name)}").limit(1).collect()
        return True
    except Exception:
        return False

print(f"Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# CELL 2 — Load and validate all score tables
T_IF = "wind-farm-a-if-scores"
T_LSTM = "wind-farm-a-lstm-scores"
T_XGB = "wind-farm-a-xgb-scores"
T_TTF = "wind-farm-a-ttf-predictions"
T_EVT = "wind-farm-a-event-info"

try:
    has_lstm = spark.catalog.tableExists(f"`{CATALOG}`.`{SCHEMA}`.`{T_LSTM}`")
except Exception:
    has_lstm = table_exists(T_LSTM)
try:
    has_ttf = spark.catalog.tableExists(f"`{CATALOG}`.`{SCHEMA}`.`{T_TTF}`")
except Exception:
    has_ttf = table_exists(T_TTF)
print(f"LSTM scores available: {has_lstm}")
print(f"TTF predictions available: {has_ttf}")

if not table_exists(T_IF):
    raise ValueError(f"Missing required table: {T_IF}")
if not table_exists(T_XGB):
    raise ValueError(f"Missing required table: {T_XGB}")
if not table_exists(T_EVT):
    raise ValueError(f"Missing required table: {T_EVT}")

# IF: dedupe at load — source table can have multiple rows per key
if_raw = spark.table(fq(T_IF))
req_if = {"asset_id", "time_stamp", "id", "train_test", "status_type_id", "if_anomaly_score"}
missing = req_if - set(if_raw.columns)
if missing:
    raise ValueError(f"{T_IF} missing columns: {missing}")

if_df = (
    if_raw.groupBy("asset_id", "time_stamp", "id")
    .agg(
        F.max("train_test").alias("train_test"),
        F.max("status_type_id").alias("status_type_id"),
        F.max("if_anomaly_score").alias("if_anomaly_score"),
    )
)

xgb_raw = spark.table(fq(T_XGB))
req_xgb = {"asset_id", "time_stamp", "id", "train_test", "status_type_id", "xgb_anomaly_prob", "xgb_fault_type"}
missing = req_xgb - set(xgb_raw.columns)
if missing:
    raise ValueError(f"{T_XGB} missing columns: {missing}")

xgb_df = xgb_raw.select(
    "asset_id", "time_stamp", "id", "train_test", "status_type_id",
    "xgb_anomaly_prob", "xgb_fault_type"
)

if has_lstm:
    lstm_raw = spark.table(fq(T_LSTM))
    if "lstm_recon_error" not in lstm_raw.columns:
        raise ValueError(f"{T_LSTM} must contain lstm_recon_error (used as lstm_anomaly_score)")
    lstm_df = lstm_raw.select(
        "asset_id",
        "time_stamp",
        "id",
        "train_test",
        "status_type_id",
        F.col("lstm_recon_error").alias("lstm_anomaly_score"),
    )
else:
    lstm_df = None

if has_ttf:
    ttf_df = spark.table(fq(T_TTF))
    if "ttf_hours" not in ttf_df.columns:
        raise ValueError(f"{T_TTF} must contain column ttf_hours")
    ttf_df = ttf_df.select("asset_id", "time_stamp", "id", F.col("ttf_hours").cast("double").alias("ttf_hours"))
else:
    ttf_df = None

evt_df = spark.table(fq(T_EVT))
req_evt = {"asset", "event_id", "event_label", "event_start_id", "event_end_id"}
missing = req_evt - set(evt_df.columns)
if missing:
    raise ValueError(f"{T_EVT} missing columns: {missing}")

def key_count(name: str, sdf):
    return sdf.select("asset_id", "time_stamp", "id").distinct().count()

kc_if = key_count(T_IF, if_df)
kc_xgb = key_count(T_XGB, xgb_df)
print("Distinct (asset_id, time_stamp, id) counts:")
print(f"  {T_IF} (deduped): {kc_if}")
print(f"  {T_XGB}: {kc_xgb}")
if has_lstm:
    print(f"  {T_LSTM}: {key_count(T_LSTM, lstm_df)}")

if kc_if != kc_xgb:
    print("WARNING: deduped IF and XGB distinct key counts differ — join will not be 1:1 on keys.")

print("\nNull summary (score columns, raw XGB / deduped IF):")
if_df.select(F.sum(F.col("if_anomaly_score").isNull().cast("int")).alias("if_nulls")).show()
xgb_df.select(F.sum(F.col("xgb_anomaly_prob").isNull().cast("int")).alias("xgb_nulls")).show()
if has_lstm:
    lstm_df.select(F.sum(F.col("lstm_anomaly_score").isNull().cast("int")).alias("lstm_nulls")).show()

# COMMAND ----------

# CELL 3 — Join all tables on [asset_id, time_stamp, id] (Spark)
# Select only score columns from side tables to avoid duplicate train_test / status_type_id.
if_scores = if_df.select("asset_id", "time_stamp", "id", F.col("if_anomaly_score"))
df_joined = xgb_df.join(if_scores, on=["asset_id", "time_stamp", "id"], how="left")

if has_lstm:
    lstm_scores = lstm_df.select(
        "asset_id", "time_stamp", "id", F.col("lstm_anomaly_score")
    )
    df_joined = df_joined.join(
        lstm_scores, on=["asset_id", "time_stamp", "id"], how="left"
    )
else:
    df_joined = df_joined.withColumn("lstm_anomaly_score", F.lit(0.0))

if has_ttf:
    df_joined = df_joined.join(
        ttf_df.select("asset_id", "time_stamp", "id", "ttf_hours"),
        on=["asset_id", "time_stamp", "id"],
        how="left",
    )
else:
    df_joined = df_joined.withColumn("ttf_hours", F.lit(None).cast("double"))

df_joined = df_joined.select(
    "asset_id",
    "time_stamp",
    "id",
    "train_test",
    "status_type_id",
    "if_anomaly_score",
    "lstm_anomaly_score",
    "xgb_anomaly_prob",
    "xgb_fault_type",
    "ttf_hours",
)

df_joined = df_joined.fillna(
    0,
    subset=[c for c in ["if_anomaly_score", "lstm_anomaly_score", "xgb_anomaly_prob"] if c in df_joined.columns],
)

df_joined = df_joined.withColumn(
    "status_type", F.col("status_type_id").cast("string")
)

pre_count = df_joined.count()
post_dedup = df_joined.dropDuplicates(["asset_id", "time_stamp", "id"]).count()
if pre_count != post_dedup:
    raise ValueError(f"Join created {pre_count - post_dedup} duplicate keys")

print(f"Joined rows (unique keys): {pre_count}")

# COMMAND ----------

# CELL 4 — Normalise scores (train rows only, no leakage)
df_pd = df_joined.toPandas()

train_mask = df_pd["train_test"] == "train"
if train_mask.sum() == 0:
    raise ValueError("No train rows for scaler fitting")

score_cols = ["if_anomaly_score"]
if has_lstm:
    score_cols.append("lstm_anomaly_score")
score_cols.append("xgb_anomaly_prob")

scaler = MinMaxScaler()
scaler.fit(df_pd.loc[train_mask, score_cols])
norm_cols = [f"{c}_norm" for c in score_cols]
df_pd[norm_cols] = scaler.transform(df_pd[score_cols].fillna(0))

print("Fitted MinMaxScaler on train rows only:", int(train_mask.sum()))
print("score_cols:", score_cols)

# COMMAND ----------

# CELL 5 — Validation-only weights & threshold; simple vs weighted ensemble
events = spark.table(fq(T_EVT)).toPandas()


def assign_window_labels(df: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    """Point-level label: 1=inside anomaly window, 0=inside normal-only window, NaN=outside labeled windows."""
    y = pd.Series(np.nan, index=df.index, dtype=float)
    for _, row in ev.iterrows():
        if row["event_label"] != "anomaly":
            continue
        m = (
            (df["asset_id"] == row["asset"])
            & (df["id"] >= row["event_start_id"])
            & (df["id"] <= row["event_end_id"])
        )
        y.loc[m] = 1.0
    for _, row in ev.iterrows():
        if row["event_label"] != "normal":
            continue
        m = (
            (df["asset_id"] == row["asset"])
            & (df["id"] >= row["event_start_id"])
            & (df["id"] <= row["event_end_id"])
        )
        m2 = m & y.isna()
        y.loc[m2] = 0.0
    return y


df_pd["y_window"] = assign_window_labels(df_pd, events)
val_mask = (df_pd["train_test"] == "train") & df_pd["y_window"].notna()
if val_mask.sum() == 0:
    raise ValueError("No validation rows (train split intersecting labeled event windows).")

val_df = df_pd.loc[val_mask].copy()
y_val = val_df["y_window"].astype(int).values

# Weights: proportional to best F1 on validation (per normalized score, threshold grid). No prediction rows.
grid = np.linspace(0.05, 0.95, 19)
f1_per_model = {}
for nc in norm_cols:
    best = 0.0
    for t in grid:
        pred = (val_df[nc].values > t).astype(int)
        best = max(best, f1_score(y_val, pred, zero_division=0))
    f1_per_model[nc] = best

raw_w = np.array([max(f1_per_model[nc], 1e-6) for nc in norm_cols], dtype=float)
active_weights = {nc: float(w) for nc, w in zip(norm_cols, raw_w / raw_w.sum())}
print("Validation F1 (per model, best threshold on grid):", f1_per_model)
print("Weights (normalized, validation-only):", active_weights)

# Version 1: simple equal-weight average
df_pd["ensemble_simple"] = df_pd[norm_cols].mean(axis=1)

# Version 2: weighted average (same norm columns; renormalize if LSTM missing)
total_w = sum(active_weights.values())
df_pd["ensemble_score"] = sum(df_pd[k] * (v / total_w) for k, v in active_weights.items())

print("\nEnsemble score summary (weighted):")
print(df_pd["ensemble_score"].describe())
print(
    "\nSimple vs Weighted correlation:\n",
    df_pd[["ensemble_simple", "ensemble_score"]].corr(),
)

# Threshold for suppression: tune on validation windows only, using simple average (conservative baseline)
best_t, best_f1 = 0.65, -1.0
for t in grid:
    # Use df_pd slice — val_df was copied before ensemble_simple existed
    pred = (df_pd.loc[val_mask, "ensemble_simple"].values > t).astype(int)
    f1 = f1_score(y_val, pred, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

ENSEMBLE_THRESHOLD = float(best_t)
print(
    f"\nValidation-chosen threshold (ensemble_simple, train windows only): {ENSEMBLE_THRESHOLD:.3f} (F1={best_f1:.3f})"
)
print("Explicit comparison: simple average vs weighted = two columns ensemble_simple vs ensemble_score (see correlation above).")

# COMMAND ----------

# CELL 6 — Alert suppression and contextual thresholds
def apply_suppression(group, threshold=None):
    if threshold is None:
        threshold = ENSEMBLE_THRESHOLD
    group = group.sort_values("time_stamp")
    above = (group["ensemble_score"] > threshold).astype(int)
    consecutive = above.rolling(3, min_periods=3).sum()
    group["alert_triggered"] = (consecutive >= 3).astype(int)
    group["threshold_used"] = threshold
    return group


df_pd = df_pd.groupby("asset_id", group_keys=False).apply(
    lambda g: apply_suppression(g, ENSEMBLE_THRESHOLD)
)

threshold_map = {0: 0.65, 1: 0.70, 2: 0.70, 3: 0.85, 4: 0.90, 5: 0.75}
st_int = pd.to_numeric(df_pd["status_type"], errors="coerce").fillna(0).astype(int)
df_pd["threshold_contextual"] = st_int.map(threshold_map).fillna(0.65)
df_pd["alert_contextual"] = (
    df_pd["ensemble_score"] > df_pd["threshold_contextual"]
).astype(int)

# COMMAND ----------

# CELL 7 — CARE score calculation
events = spark.table(fq(T_EVT)).toPandas()

care_rows = []
for _, event in events.iterrows():
    aid = event["asset"]
    window_rows = df_pd[
        (df_pd["asset_id"] == aid)
        & (df_pd["id"] >= event["event_start_id"])
        & (df_pd["id"] <= event["event_end_id"])
    ]
    if len(window_rows) == 0:
        continue

    is_anomaly = event["event_label"] == "anomaly"
    alerts_in_window = int(window_rows["alert_triggered"].sum())

    coverage = (alerts_in_window / len(window_rows)) if is_anomaly else None

    first_alert_rows = window_rows[window_rows["alert_triggered"] == 1]
    if is_anomaly and len(first_alert_rows) > 0:
        event_end_time = window_rows["time_stamp"].max()
        first_alert_time = first_alert_rows["time_stamp"].min()
        earliness_h = (
            pd.to_datetime(event_end_time) - pd.to_datetime(first_alert_time)
        ).total_seconds() / 3600
    else:
        earliness_h = None

    desc = event["event_description"] if "event_description" in event.index else ""
    care_rows.append(
        {
            "event_id": event["event_id"],
            "asset_id": aid,
            "event_label": event["event_label"],
            "fault_type": desc if pd.notna(desc) else "",
            "window_rows": len(window_rows),
            "alerts_triggered": alerts_in_window,
            "coverage": coverage,
            "earliness_h": earliness_h,
        }
    )

care_df = pd.DataFrame(care_rows)

# Reliability (false alarm rate) on normal-labeled windows
normal_evt = events[events["event_label"] == "normal"][
    ["asset", "event_start_id", "event_end_id"]
].rename(columns={"asset": "asset_id"})

nr_list = []
for _, er in normal_evt.iterrows():
    m = (
        (df_pd["asset_id"] == er["asset_id"])
        & (df_pd["id"] >= er["event_start_id"])
        & (df_pd["id"] <= er["event_end_id"])
    )
    nr_list.append(df_pd.loc[m])
normal_rows = pd.concat(nr_list, axis=0) if nr_list else pd.DataFrame(columns=df_pd.columns)

if len(normal_rows) == 0:
    reliability = 1.0
else:
    reliability = 1 - (normal_rows["alert_triggered"].sum() / max(len(normal_rows), 1))

print("=== CARE Score Summary ===")
cov_mean = care_df.loc[care_df["event_label"] == "anomaly", "coverage"].mean()
earl_mean = care_df["earliness_h"].mean()
print(f"Coverage  (mean over anomaly windows): {cov_mean:.3f}")
print(f"Earliness (mean hours, anomaly windows with alerts): {earl_mean:.1f}h")
print(f"Reliability (1 - false alarm rate in normal windows): {reliability:.3f}")
print("\nCoverage by fault type:")
print(
    care_df.loc[care_df["event_label"] == "anomaly"]
    .groupby("fault_type")["coverage"]
    .mean()
)

if HAS_MLFLOW:
    with mlflow.start_run(run_name="care_ensemble"):
        mlflow.log_metric("care_coverage", float(cov_mean) if pd.notna(cov_mean) else 0.0)
        mlflow.log_metric(
            "care_earliness_h", float(earl_mean) if pd.notna(earl_mean) else 0.0
        )
        mlflow.log_metric("care_reliability", float(reliability))
else:
    print("mlflow not installed — skipped logging.")

# COMMAND ----------

# CELL 8 — Save CARE summary as intermediate artifact
_tbl_care = f"`{CATALOG}`.`{SCHEMA}`.`care-score-summary`"
spark.createDataFrame(care_df).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(_tbl_care)

# COMMAND ----------

# CELL 9 — Output schema validation then save
required_alerts = [
    "asset_id",
    "time_stamp",
    "id",
    "train_test",
    "status_type",
    "if_anomaly_score",
    "lstm_anomaly_score",
    "xgb_anomaly_prob",
    "ensemble_score",
    "alert_triggered",
    "alert_contextual",
    "threshold_used",
]
for col in required_alerts:
    if col not in df_pd.columns:
        raise ValueError(f"Output missing column: {col}")

out_alerts = df_pd[
    [
        "asset_id",
        "time_stamp",
        "id",
        "train_test",
        "status_type",
        "if_anomaly_score",
        "lstm_anomaly_score",
        "xgb_anomaly_prob",
        "ensemble_score",
        "alert_triggered",
        "alert_contextual",
        "threshold_used",
    ]
].copy()

_tbl_alerts = f"`{CATALOG}`.`{SCHEMA}`.`wind-farm-a-ensemble-alerts`"
spark.createDataFrame(out_alerts).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(_tbl_alerts)

alert_summary = df_pd.loc[df_pd["alert_triggered"] == 1].copy()
alert_summary["alert_id"] = (
    alert_summary["asset_id"].astype(str)
    + "_"
    + alert_summary["time_stamp"].astype(str)
)

if "ttf_hours" not in alert_summary.columns:
    alert_summary["ttf_hours"] = np.nan
else:
    alert_summary["ttf_hours"] = pd.to_numeric(alert_summary["ttf_hours"], errors="coerce")

ttf_fill = alert_summary["ttf_hours"].fillna(999)
alert_summary["urgency"] = np.where(
    ttf_fill < 24,
    "CRITICAL",
    np.where(ttf_fill < 72, "HIGH", "MONITOR"),
)

summary_cols = [
    "alert_id",
    "asset_id",
    "time_stamp",
    "ensemble_score",
    "xgb_fault_type",
    "ttf_hours",
    "urgency",
]
for c in summary_cols:
    if c not in alert_summary.columns:
        raise ValueError(f"alert_summary missing column: {c}")

alert_out = alert_summary[summary_cols].rename(columns={"time_stamp": "alert_time"})

_tbl_summary = f"`{CATALOG}`.`{SCHEMA}`.`wind-farm-a-alert-summary`"
spark.createDataFrame(alert_out).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(_tbl_summary)

print("Saved. Final alert count:", spark.table(_tbl_summary).count())