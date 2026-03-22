# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 6 — Explainability (scoped)
# MAGIC
# MAGIC - **Top 5** highest `ensemble_score` alerts → SHAP waterfall each (logged to MLflow).
# MAGIC - **One case-study turbine**: prefer **largest `ttf_hours` at alert** (optimistic TTF ⇒ risk of miss); if TTF unavailable, fallback to **turbine with most alerts**.
# MAGIC - **No full SHAP timelines** for every timestep across all turbines; case study uses **one** 120h window only.
# MAGIC - Row-level SHAP for **all alert rows** (CELL 5) can be heavy — set `MAX_ALERTS_FOR_ROW_SHAP` to cap if needed.
# MAGIC
# MAGIC **Prereqs:** `04_care_ensemble`, feature table populated, registered MLflow model `xgb-anomaly-detector` (Staging), cluster with `shap`, `xgboost`, `mlflow`.

# COMMAND ----------

# CELL 1 — Imports and MLflow experiment
import json
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from pyspark.sql import functions as F

warnings.filterwarnings("ignore", category=UserWarning)

CATALOG = "workspace"
SCHEMA = "wind-turbine-silver"


def fq(name: str) -> str:
    return f"`{CATALOG}`.`{SCHEMA}`.`{name}`"


# Per mlflow-onboarding: use a dedicated experiment for traditional ML artifacts
mlflow.set_experiment("/DSMLC-Final-Comp-2026/explainability")
print("MLflow experiment:", mlflow.get_experiment_by_name("/DSMLC-Final-Comp-2026/explainability").experiment_id)

# COMMAND ----------

# CELL 2 — Load and validate
MAX_ALERTS_FOR_ROW_SHAP = None  # e.g. 500 for smoke test; None = all alert rows

alerts_spark = spark.table(fq("wind-farm-a-ensemble-alerts"))
feats_spark = spark.table(fq("wind-farm-a-features"))
xgb_spark = spark.table(fq("wind-farm-a-xgb-scores")).select(
    "asset_id",
    "time_stamp",
    "id",
    F.col("xgb_fault_type").alias("xgb_fault_type"),
    F.col("xgb_anomaly_prob").alias("xgb_prob_join"),
)

has_ttf = True
try:
    spark.sql(f"DESCRIBE TABLE {fq('wind-farm-a-ttf-predictions')}").limit(1).collect()
    ttf_spark = spark.table(fq("wind-farm-a-ttf-predictions")).select(
        "asset_id", "time_stamp", "id", F.col("ttf_hours").cast("double").alias("ttf_hours")
    )
except Exception:
    has_ttf = False
    ttf_spark = None
    print("TTF table not found — ttf_hours will default to 999 in outputs.")

for name, tbl in [("alerts", alerts_spark), ("features", feats_spark)]:
    count = tbl.count()
    if count == 0:
        raise ValueError(
            f"{name} table is empty — check upstream notebooks ran successfully"
        )
    print(f"{name}: {count} rows")

# Non-model features / keys / labels (exclude from XGB SHAP input)
EXCLUDE_COLS = {
    "asset_id",
    "time_stamp",
    "id",
    "train_test",
    "status_type",
    "status_type_id",
    "event_id",
    "farm",
    "ensemble_score",
    "alert_triggered",
    "if_anomaly_score",
    "lstm_anomaly_score",
    "xgb_anomaly_prob",
    "alert_contextual",
    "threshold_used",
    "xgb_fault_type",
    "xgb_prob_join",
    "ttf_hours",
    # Leakage / supervision targets from feature engineering (UC schema)
    "next_anomaly_event_id",
    "hours_to_next_anomaly",
    "risk_score",
    "is_usable_supervised_next",
    "hours_to_anomaly_linked_downtime",
    "risk_score_anomaly_linked_downtime",
    "hours_to_next_operator_warning",
    "risk_score_operator_warning",
    "in_anomaly_window",
    "if_fit_eligible",
}

feature_cols = [c for c in feats_spark.columns if c not in EXCLUDE_COLS]
if len(feature_cols) == 0:
    raise ValueError("No feature columns found in features table")

feat_subset = feats_spark.select(["asset_id", "time_stamp", "id"] + feature_cols)

joined = (
    alerts_spark.filter(F.col("alert_triggered") == 1)
    .join(feat_subset, on=["asset_id", "time_stamp", "id"], how="left")
    .join(xgb_spark, on=["asset_id", "time_stamp", "id"], how="left")
)
if has_ttf:
    joined = joined.join(ttf_spark, on=["asset_id", "time_stamp", "id"], how="left")
else:
    joined = joined.withColumn("ttf_hours", F.lit(None).cast("double"))

df_alerts = joined.toPandas()
if MAX_ALERTS_FOR_ROW_SHAP is not None:
    df_alerts = df_alerts.nlargest(MAX_ALERTS_FOR_ROW_SHAP, "ensemble_score").reset_index(
        drop=True
    )
    print(f"Capped to top {MAX_ALERTS_FOR_ROW_SHAP} alerts by ensemble_score for row SHAP.")

print(f"Alert rows loaded: {len(df_alerts)}")
print(f"Feature columns for model: {len(feature_cols)}")

# COMMAND ----------

# CELL 3 — Load XGBoost model from MLflow
MODEL_URI = "models:/xgb-anomaly-detector/Staging"
xgb_l1 = mlflow.xgboost.load_model(MODEL_URI)

X_test = df_alerts[feature_cols].head(1).fillna(0)
test_pred = xgb_l1.predict_proba(X_test)
print(f"Model loaded OK from {MODEL_URI}. Test pred shape: {test_pred.shape}")

# COMMAND ----------

# CELL 4 — SHAP waterfall for top 5 alerts only
top5 = df_alerts.nlargest(5, "ensemble_score").copy()
X_top5 = top5[feature_cols].fillna(0)

explainer = shap.TreeExplainer(xgb_l1)
shap_top5 = explainer(X_top5)

with mlflow.start_run(run_name="shap_top5_alerts"):
    for i, (idx, row) in enumerate(top5.iterrows()):
        plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_top5[i], max_display=10, show=False)
        plt.title(f"SHAP — asset {row['asset_id']} {row['time_stamp']}")
        plt.tight_layout()
        fname = f"shap_top{i+1}_asset_{row['asset_id']}.png"
        path = f"/tmp/{fname}"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(path)
        plt.close()
    print("Top 5 SHAP waterfall plots logged to MLflow")

# COMMAND ----------

# CELL 5 — Top SHAP sensors JSON for each alert row (batched)
X_all_alerts = df_alerts[feature_cols].fillna(0)
explainer_all = shap.TreeExplainer(xgb_l1)
shap_all = explainer_all(X_all_alerts)


def get_top_shap(shap_row, feature_names, n=3):
    v = np.asarray(shap_row.values).ravel()
    idx = np.argsort(np.abs(v))[-n:][::-1]
    total = float(np.sum(np.abs(v)) + 1e-12)
    out = []
    for j in idx:
        out.append(
            {
                "sensor": feature_names[j],
                "contribution_pct": round(float(np.abs(v[j])) / total * 100, 1),
            }
        )
    return json.dumps(out)


df_alerts["top_shap_sensors"] = [
    get_top_shap(shap_all[i], feature_cols) for i in range(len(df_alerts))
]
print("Computed top_shap_sensors for each alert row.")

# COMMAND ----------

# CELL 6 — Counterfactual text per alert (top sensor only)
def counterfactual_explanation(model, x_row, top_sensor_col, feature_cols, reduction=0.20):
    x_orig = x_row.fillna(0).values.astype(np.float64)
    x_mod = x_orig.copy()
    idx = feature_cols.index(top_sensor_col)
    x_mod[idx] *= 1 - reduction
    p_orig = float(model.predict_proba([x_orig])[0][1])
    p_mod = float(model.predict_proba([x_mod])[0][1])
    drop = round((p_orig - p_mod) / max(p_orig, 0.001) * 100, 1)
    name = top_sensor_col.replace("_avg", "").replace("_", " ")
    return (
        f"If {name} had been {int(reduction * 100)}% lower, "
        f"estimated failure probability drops ~{drop}% relative to baseline."
    )


cf_texts = []
for idx, row in df_alerts.iterrows():
    try:
        sensors = json.loads(row["top_shap_sensors"])
        top_s = sensors[0]["sensor"]
        cf = counterfactual_explanation(
            xgb_l1, df_alerts.loc[idx, feature_cols], top_s, feature_cols
        )
    except Exception as e:
        cf = f"Counterfactual unavailable: {e}"
    cf_texts.append(cf)

df_alerts["counterfactual_text"] = cf_texts

# COMMAND ----------

# CELL 7 — Case study: one turbine (highest TTF at alert → miss risk; else most alerts)
if df_alerts["ttf_hours"].notna().any():
    imax = df_alerts["ttf_hours"].idxmax()
    case_turbine = int(df_alerts.loc[imax, "asset_id"])
    print(
        f"Case study turbine (max ttf_hours among alerts): {case_turbine} "
        f"(ttf_hours={df_alerts.loc[imax, 'ttf_hours']})"
    )
else:
    case_turbine = int(df_alerts.groupby("asset_id").size().idxmax())
    print(
        f"Case study turbine (no TTF — fallback to most alerts): {case_turbine}"
    )

first_alert_time = pd.to_datetime(
    df_alerts.loc[df_alerts["asset_id"] == case_turbine, "time_stamp"].min()
)
start_ts = first_alert_time - pd.Timedelta(hours=120)

case_rows = (
    feats_spark.filter(F.col("asset_id") == case_turbine)
    .filter(F.col("time_stamp") >= F.lit(start_ts))
    .filter(F.col("time_stamp") <= F.lit(first_alert_time))
    .orderBy("time_stamp")
    .toPandas()
)

if len(case_rows) > 0:
    X_case = case_rows[feature_cols].fillna(0)
    shap_c = explainer_all(X_case)
    mean_abs = np.abs(shap_c.values).mean(axis=0)
    top8_idx = np.argsort(mean_abs)[-8:][::-1]
    top8_cols = [feature_cols[i] for i in top8_idx]
    shap_df = pd.DataFrame(shap_c.values[:, top8_idx], columns=top8_cols)
    shap_df.index = case_rows["time_stamp"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(shap_df.T, cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "SHAP value"})
    ax.set_title(f"Sensor SHAP timeline (120h before first alert) — asset {case_turbine}")
    ax.set_xlabel("Time step (rows in window)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    tpath = f"/tmp/timeline_asset_{case_turbine}.png"
    plt.savefig(tpath, dpi=150, bbox_inches="tight")
    with mlflow.start_run(run_name="shap_case_study_timeline"):
        mlflow.log_artifact(tpath)
    plt.close()
    print(f"Timeline saved: {tpath}")
else:
    print("No feature rows in 120h pre-alert window — skip heatmap.")

# COMMAND ----------

# CELL 8 — Build alert records, validate schema, save Delta table
FARM_NAME = "wind-farm-a"


def safe_ttf(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return 999
    return int(round(float(v)))


records = []
for _, row in df_alerts.iterrows():
    fault = row.get("xgb_fault_type") or "unknown"
    if isinstance(fault, float) and np.isnan(fault):
        fault = "unknown"
    prob = row.get("xgb_anomaly_prob", row.get("xgb_prob_join", 0))
    conf = round(float(prob) * 100, 1) if pd.notna(prob) else 0.0
    ttf = safe_ttf(row.get("ttf_hours"))
    urgency = "CRITICAL" if ttf < 24 else "HIGH" if ttf < 72 else "MONITOR"
    ft = str(fault).lower() if fault else "unknown"
    records.append(
        {
            "alert_id": f"{row['asset_id']}_{row['time_stamp']}",
            "turbine_id": int(row["asset_id"]),
            "farm": FARM_NAME,
            "alert_time": str(row["time_stamp"]),
            "ensemble_score": round(float(row["ensemble_score"]), 3),
            "fault_type": fault,
            "confidence_pct": conf,
            "ttf_hours": ttf,
            "urgency": urgency,
            "top_shap_sensors": row.get("top_shap_sensors", "[]"),
            "counterfactual_text": row.get("counterfactual_text", ""),
            "recommended_action": (
                f"Inspect {ft} subsystem within {max(6, ttf // 2)} hours; validate top SHAP sensors."
            ),
        }
    )

records_df = pd.DataFrame(records)

required_out = [
    "alert_id",
    "turbine_id",
    "farm",
    "alert_time",
    "ensemble_score",
    "fault_type",
    "confidence_pct",
    "ttf_hours",
    "urgency",
    "top_shap_sensors",
    "counterfactual_text",
    "recommended_action",
]
for col in required_out:
    if col not in records_df.columns:
        raise ValueError(f"Output missing column: {col}")

_tbl = f"`{CATALOG}`.`{SCHEMA}`.`wind-farm-a-alert-records`"
spark.createDataFrame(records_df).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(_tbl)
print(f"Saved {len(records_df)} rows to {_tbl}")