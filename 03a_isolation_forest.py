# Databricks notebook source
# MAGIC %md
# MAGIC # 03a — Isolation Forest (Farm A)
# MAGIC Per-turbine anomaly detection with MLflow experiment tracking.
# MAGIC
# MAGIC **Output table contract** — `wind-farm-a-if-scores` must include:
# MAGIC `asset_id`, `time_stamp`, `id`, `train_test`, `status_type_id`,
# MAGIC `if_anomaly_score` (float), `if_anomaly_flag` (int 0/1)

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELL 1 — Imports and experiment setup

import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from pyspark.sql import functions as F
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/" + spark.sql("SELECT current_user()").first()[0] + "/DSMLC-Final-Comp-2026-isolation-forest")
mlflow.sklearn.autolog(disable=True)

CONTAMINATION_VALUES = [0.01, 0.03, 0.05]

CATALOG = "wind-turbine-silver"

# COMMAND ----------

# CELL 2 — Load and validate

df_spark = spark.table(f"`{CATALOG}`.`wind-farm-a-features`")

required = ['asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id']
missing = [c for c in required if c not in df_spark.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

EXCLUDE_COLS = {
    'asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id',
    'event_id', 'farm', 'event_label', 'event_description',
}
feature_cols = [c for c in df_spark.columns if c not in EXCLUDE_COLS]

if len(feature_cols) == 0:
    raise ValueError(
        "No feature columns found — check that 02_feature_engineering ran first"
    )

print(f"Feature count: {len(feature_cols)}")
print(f"Turbine count: {df_spark.select('asset_id').distinct().count()}")
df_spark.groupBy('train_test', 'status_type_id').count().orderBy(
    'train_test', 'status_type_id'
).show()

# COMMAND ----------

# CELL 3 — Training loop (one model per turbine, nested MLflow runs)

turbines = sorted(
    [r.asset_id for r in df_spark.select('asset_id').distinct().collect()]
)
print(f"Training Isolation Forest for {len(turbines)} turbines ...\n")

all_scores = []

with mlflow.start_run(run_name="IF_Farm_A_parent"):
    mlflow.log_param("n_turbines", len(turbines))
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("contamination_values", CONTAMINATION_VALUES)

    for idx, asset_id in enumerate(turbines):
        turbine_pd = (
            df_spark
            .filter(F.col('asset_id') == asset_id)
            .orderBy('time_stamp')
            .toPandas()
        )

        train_mask = turbine_pd['train_test'] == 'train'
        if train_mask.sum() == 0:
            raise ValueError(f"Turbine {asset_id} has no train rows")

        X_train = turbine_pd.loc[train_mask, feature_cols].fillna(0).values
        X_all   = turbine_pd[feature_cols].fillna(0).values

        best_model = None
        best_cont  = CONTAMINATION_VALUES[1]
        best_score = np.inf

        with mlflow.start_run(run_name=f"IF_{asset_id}", nested=True):
            for cont in CONTAMINATION_VALUES:
                model = IsolationForest(
                    contamination=cont, random_state=42, n_jobs=-1
                )
                model.fit(X_train)
                val_scores = -model.decision_function(X_train)
                mean_val = float(val_scores.mean())
                mlflow.log_metric(f"mean_score_cont_{cont}", mean_val)

                if mean_val < best_score:
                    best_score = mean_val
                    best_cont  = cont
                    best_model = model

            scores = -best_model.decision_function(X_all)

            mlflow.sklearn.log_model(best_model, "isolation_forest")
            mlflow.log_param("asset_id", asset_id)
            mlflow.log_param("contamination", best_cont)
            mlflow.log_param("train_rows", int(train_mask.sum()))
            mlflow.log_metric("mean_score_all", float(scores.mean()))

        turbine_pd['if_anomaly_score'] = scores
        turbine_pd['if_anomaly_flag'] = (
            scores > np.percentile(scores, 95)
        ).astype(int)

        all_scores.append(
            turbine_pd[
                ['asset_id', 'time_stamp', 'id', 'train_test',
                 'status_type_id', 'if_anomaly_score', 'if_anomaly_flag']
            ]
        )

        print(
            f"  [{idx+1}/{len(turbines)}] Turbine {asset_id}: "
            f"cont={best_cont}, mean_score={scores.mean():.4f}, "
            f"flagged={int(turbine_pd['if_anomaly_flag'].sum())}"
        )

df_scored = pd.concat(all_scores, ignore_index=True)
print(f"\nDone. Total scored rows: {len(df_scored):,}")

# COMMAND ----------

# CELL 4 — Validation against event labels

events = spark.table(f"`{CATALOG}`.`wind-farm-a-event-info`").toPandas()
events = events.rename(columns={'asset': 'asset_id'})

print(f"Event table: {len(events)} events")
print(f"  Labels: {events['event_label'].value_counts().to_dict()}\n")

df_eval = df_scored.merge(
    events[['asset_id', 'event_id', 'event_label',
            'event_start_id', 'event_end_id', 'event_description']],
    on='asset_id', how='inner',
)
df_eval['in_event'] = (
    (df_eval['id'] >= df_eval['event_start_id'])
    & (df_eval['id'] <= df_eval['event_end_id'])
)
df_eval = df_eval[df_eval['in_event']].copy()

print(f"Rows inside event windows: {len(df_eval):,}\n")

print("Mean IF score by event label:")
print(df_eval.groupby('event_label')['if_anomaly_score'].agg(
    ['mean', 'std', 'count']
))

anomaly_rows = df_eval[df_eval['event_label'] == 'anomaly']
if not anomaly_rows.empty:
    print("\nMean IF score by fault type:")
    print(
        anomaly_rows
        .groupby('event_description')['if_anomaly_score']
        .mean()
        .sort_values(ascending=False)
    )

# COMMAND ----------

# CELL 7 — IF Model Improvement & Diagnostics

print("=" * 70)
print("CELL 7 — IF Model Improvement & Diagnostics")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# Re-load features (df_scored from Cell 3 has metadata + scores only;
# df_spark from Cell 2 is still in memory and has all feature columns)
# ═══════════════════════════════════════════════════════════════════════

print("Re-loading feature data from df_spark ...")
df_features_pd = df_spark.toPandas()

df_full = df_scored.merge(
    df_features_pd[['asset_id', 'id'] + feature_cols],
    on=['asset_id', 'id'],
    how='left',
)
assert len(df_full) == len(df_scored), \
    f"Merge changed row count: {len(df_scored)} → {len(df_full)}"
print(f"Merged: {len(df_full):,} rows × {len(feature_cols)} features")

# ═══════════════════════════════════════════════════════════════════════
# Build event-window masks (kept separate from df_scored / df_full
# so temporary labels don't leak into the saved Delta table)
# ═══════════════════════════════════════════════════════════════════════

anomaly_mask = pd.Series(False, index=df_full.index)
normal_mask  = pd.Series(False, index=df_full.index)
event_desc   = pd.Series(None,  index=df_full.index, dtype=object)

for _, evt in events.iterrows():
    row_mask = (
        (df_full['asset_id'] == int(evt['asset_id'])) &
        (df_full['id'] >= int(evt['event_start_id'])) &
        (df_full['id'] <= int(evt['event_end_id']))
    )
    if evt['event_label'] == 'anomaly':
        anomaly_mask |= row_mask
        if pd.notna(evt.get('event_description')):
            event_desc[row_mask] = evt['event_description']
    else:
        normal_mask |= row_mask

print(f"\nEvent window distribution:")
print(f"  Anomaly event rows: {anomaly_mask.sum():,}")
print(f"  Normal event rows:  {normal_mask.sum():,}")
print(f"  Outside events:     {(~anomaly_mask & ~normal_mask).sum():,}")

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — Feature importance via mean score gap
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 1 — Feature importance via mean score gap")
print("=" * 70)

anomaly_feat = df_full.loc[anomaly_mask, feature_cols]
normal_feat  = df_full.loc[normal_mask,  feature_cols]

sep_records = []
for feat in feature_cols:
    a_mean = anomaly_feat[feat].mean()
    n_mean = normal_feat[feat].mean()
    sep_records.append({
        'feature': feat,
        'anomaly_mean': a_mean,
        'normal_mean': n_mean,
        'separation_score': a_mean - n_mean,
    })

feature_sep_df = (
    pd.DataFrame(sep_records)
    .sort_values('separation_score', ascending=False)
    .reset_index(drop=True)
)

print("\nTop 20 most discriminating features (by separation_score):")
print(feature_sep_df.head(20).to_string(index=False))

feature_sep_df.to_csv("/tmp/if_feature_separation.csv", index=False)

runs = mlflow.search_runs(
    filter_string="tags.mlflow.runName = 'IF_Farm_A_parent'",
    max_results=1,
)
parent_run_id = runs.iloc[0].run_id
print(f"\nFound parent MLflow run: {parent_run_id}")

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_artifact("/tmp/if_feature_separation.csv")
print("Logged feature_sep_df as CSV artifact to MLflow.")

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — Retrain using only top discriminating features
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 2 — Retrain using only top discriminating features")
print("=" * 70)

N_TOP = min(50, len(feature_cols))
top_feature_cols = feature_sep_df.head(N_TOP)['feature'].tolist()
print(f"\nUsing top {len(top_feature_cols)} features for v2 model.")

df_full['if_anomaly_score_v2'] = np.nan
df_full['if_anomaly_flag_v2']  = 0

for turbine_id in sorted(df_full['asset_id'].unique()):
    turbine_mask = df_full['asset_id'] == turbine_id

    train_mask = (
        turbine_mask
        & (df_full['status_type_id'] == 0)
        & (df_full['train_test'] == 'train')
    )

    X_train = df_full.loc[train_mask, top_feature_cols].dropna()
    if len(X_train) == 0:
        print(f"  Turbine {turbine_id}: no training data — skipping")
        continue

    iso_v2 = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    iso_v2.fit(X_train)

    X_all  = df_full.loc[turbine_mask, top_feature_cols].fillna(0)
    scores = -iso_v2.decision_function(X_all)

    df_full.loc[turbine_mask, 'if_anomaly_score_v2'] = scores

    threshold_95 = np.percentile(scores, 95)
    df_full.loc[turbine_mask, 'if_anomaly_flag_v2'] = (
        scores > threshold_95
    ).astype(int)

    print(f"  Turbine {turbine_id}: trained on {len(X_train):,} rows, "
          f"scored {turbine_mask.sum():,} rows")

print(f"\nv2 anomaly flag counts (initial per-turbine 95th percentile):")
print(df_full['if_anomaly_flag_v2'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — Re-validate and compare v1 vs v2
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 3 — Re-validate and compare v1 vs v2")
print("=" * 70)

v1_normal_mean  = df_full.loc[normal_mask,  'if_anomaly_score'].mean()
v1_anomaly_mean = df_full.loc[anomaly_mask, 'if_anomaly_score'].mean()
v1_sep = v1_anomaly_mean - v1_normal_mean
v1_dir = "OK" if v1_sep > 0 else "WRONG"

v2_normal_mean  = df_full.loc[normal_mask,  'if_anomaly_score_v2'].mean()
v2_anomaly_mean = df_full.loc[anomaly_mask, 'if_anomaly_score_v2'].mean()
v2_sep = v2_anomaly_mean - v2_normal_mean
v2_dir = "OK" if v2_sep > 0 else "WRONG"

print(f"\n{'Version':<10} {'Normal mean':>12} {'Anomaly mean':>13} "
      f"{'Separation':>11} {'Direction':>10}")
print("-" * 60)
print(f"{'v1':<10} {v1_normal_mean:>12.4f} {v1_anomaly_mean:>13.4f} "
      f"{v1_sep:>11.4f} {v1_dir:>10}")
print(f"{'v2':<10} {v2_normal_mean:>12.4f} {v2_anomaly_mean:>13.4f} "
      f"{v2_sep:>11.4f} {v2_dir:>10}")

print("\nPer-fault-type mean v2 scores:")
anomaly_with_desc = df_full.loc[anomaly_mask].copy()
anomaly_with_desc['event_description'] = event_desc[anomaly_mask].values
fault_scores = (
    anomaly_with_desc
    .groupby('event_description')['if_anomaly_score_v2']
    .mean()
    .sort_values(ascending=False)
)
print(fault_scores.to_string())

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4 — Threshold tuning on validation windows
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 4 — Threshold tuning on validation windows")
print("=" * 70)

event_mask = anomaly_mask | normal_mask
event_rows = df_full[event_mask]
event_anomaly = anomaly_mask[event_mask]
event_normal  = normal_mask[event_mask]

percentiles_to_test = [90, 92, 94, 95, 96, 97, 98, 99]
threshold_results = []

for pct in percentiles_to_test:
    threshold = np.percentile(
        df_full['if_anomaly_score_v2'].dropna(), pct
    )
    above = event_rows['if_anomaly_score_v2'] > threshold

    tp = int((above & event_anomaly).sum())
    fp = int((above & event_normal).sum())
    fn = int((~above & event_anomaly).sum())
    tn = int((~above & event_normal).sum())

    detection_rate   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    threshold_results.append({
        'percentile': pct,
        'threshold': threshold,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
    })

threshold_df = pd.DataFrame(threshold_results)

print(f"\n{'Percentile':>10} {'Threshold':>10} {'Detection rate':>15} "
      f"{'False alarm rate':>17}")
print("-" * 56)
for _, row in threshold_df.iterrows():
    print(f"{row['percentile']:>10.0f} {row['threshold']:>10.4f} "
          f"{row['detection_rate']:>15.4f} {row['false_alarm_rate']:>17.4f}")

candidates = threshold_df[threshold_df['false_alarm_rate'] < 0.10]
if len(candidates) > 0:
    best_idx = candidates['detection_rate'].idxmax()
    BEST_PERCENTILE = int(candidates.loc[best_idx, 'percentile'])
else:
    threshold_df['_balance'] = (
        threshold_df['detection_rate'] - threshold_df['false_alarm_rate']
    )
    best_idx = threshold_df['_balance'].idxmax()
    BEST_PERCENTILE = int(threshold_df.loc[best_idx, 'percentile'])
    threshold_df.drop(columns=['_balance'], inplace=True)

print(f"\nRecommended threshold percentile: {BEST_PERCENTILE}")

threshold_df.to_csv("/tmp/if_threshold_comparison.csv", index=False)
with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_artifact("/tmp/if_threshold_comparison.csv")
print("Logged threshold comparison CSV to MLflow.")

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5 — Update if_anomaly_flag_v2 using best threshold
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 5 — Update if_anomaly_flag_v2 using best threshold")
print("=" * 70)

best_threshold = np.percentile(
    df_full['if_anomaly_score_v2'].dropna(), BEST_PERCENTILE
)
df_full['if_anomaly_flag_v2'] = (
    df_full['if_anomaly_score_v2'] > best_threshold
).astype(int)

print(f"\nBest threshold (percentile {BEST_PERCENTILE}): {best_threshold:.6f}")
print(f"\nUpdated flag counts:")
print(df_full['if_anomaly_flag_v2'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 6 — Log v2 summary metrics to MLflow
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 6 — Log v2 summary metrics to MLflow")
print("=" * 70)

best_row = threshold_df[threshold_df['percentile'] == BEST_PERCENTILE].iloc[0]
detection_rate_at_best = best_row['detection_rate']
false_alarm_at_best    = best_row['false_alarm_rate']

normal_mean_v2  = v2_normal_mean
anomaly_mean_v2 = v2_anomaly_mean
separation_v2   = v2_sep

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("v2_normal_mean_score",  normal_mean_v2)
    mlflow.log_metric("v2_anomaly_mean_score", anomaly_mean_v2)
    mlflow.log_metric("v2_separation_score",   separation_v2)
    mlflow.log_metric("v2_best_percentile",    BEST_PERCENTILE)
    mlflow.log_metric("v2_detection_rate",     detection_rate_at_best)
    mlflow.log_metric("v2_false_alarm_rate",   false_alarm_at_best)
    mlflow.log_param("v2_top_feature_count",   len(top_feature_cols))

print(f"\nLogged to MLflow parent run '{parent_run_id}':")
print(f"  v2_normal_mean_score:  {normal_mean_v2:.6f}")
print(f"  v2_anomaly_mean_score: {anomaly_mean_v2:.6f}")
print(f"  v2_separation_score:   {separation_v2:.6f}")
print(f"  v2_best_percentile:    {BEST_PERCENTILE}")
print(f"  v2_detection_rate:     {detection_rate_at_best:.4f}")
print(f"  v2_false_alarm_rate:   {false_alarm_at_best:.4f}")
print(f"  v2_top_feature_count:  {len(top_feature_cols)}")

# ═══════════════════════════════════════════════════════════════════════
# Copy v2 columns back to df_scored (which only has metadata + v1 scores)
# ═══════════════════════════════════════════════════════════════════════

df_scored['if_anomaly_score_v2'] = df_full['if_anomaly_score_v2'].values
df_scored['if_anomaly_flag_v2']  = df_full['if_anomaly_flag_v2'].values

print("\n" + "=" * 70)
print("CELL 7 complete.")
print("df_scored now contains if_anomaly_score_v2 and if_anomaly_flag_v2.")
print("=" * 70)

# COMMAND ----------

# CELL 5 — Output schema validation

required_out = [
    'asset_id', 'time_stamp', 'id', 'train_test',
    'status_type_id',
    'if_anomaly_score',    'if_anomaly_flag',
    'if_anomaly_score_v2', 'if_anomaly_flag_v2',
]
for col in required_out:
    if col not in df_scored.columns:
        raise ValueError(f"Output missing required column: {col}")

print("Output schema validated.")
print(df_scored.dtypes)
print(f"\nTotal rows: {len(df_scored):,}")
print(f"\nNull counts:\n{df_scored.isnull().sum()}")

# COMMAND ----------

# CELL 6 — Save to Delta

(spark.createDataFrame(df_scored)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`wind-farm-a-if-scores`"))

saved = spark.table(f"`{CATALOG}`.`wind-farm-a-if-scores`")
print(f"Saved rows: {saved.count():,}")
saved.groupBy('if_anomaly_flag').count().show()

print("Columns in saved table:", saved.columns)
assert 'if_anomaly_score_v2' in saved.columns, \
    "v2 score column missing from saved table"
assert 'if_anomaly_flag_v2' in saved.columns, \
    "v2 flag column missing from saved table"
print("v2 columns confirmed in Delta table.")
saved.groupBy('if_anomaly_flag_v2').count().show()
