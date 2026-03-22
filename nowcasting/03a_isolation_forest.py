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
import gc
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

_META_COLS = ['asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id']

print(f"Feature count: {len(feature_cols)}")
print(f"Turbine count: {df_spark.select('asset_id').distinct().count()}")
df_spark.groupBy('train_test', 'status_type_id').count().orderBy(
    'train_test', 'status_type_id'
).show()

# COMMAND ----------

# CELL 3 — Training loop (one model per turbine, nested MLflow runs)
#
# Memory safety: only pull needed columns to the driver, use float32, avoid
# n_jobs=-1 (fork spikes), and gc.collect() each turbine — prevents OOM /
# "Python kernel is unresponsive" on large feature matrices.

_IF_COLS = _META_COLS + feature_cols

turbines = sorted(
    [r.asset_id for r in df_spark.select('asset_id').distinct().collect()]
)
print(f"Training Isolation Forest for {len(turbines)} turbines ...\n")

# Load anomaly event windows — rows inside these must be excluded
# from training so the model learns truly normal behaviour only.
_events_pd = spark.table(f"`{CATALOG}`.`wind-farm-a-event-info`").toPandas()
_events_pd = _events_pd.rename(columns={'asset': 'asset_id'})
_anomaly_events = _events_pd.loc[
    _events_pd['event_label'] == 'anomaly',
    ['asset_id', 'event_start_id', 'event_end_id'],
].copy()
print(f"Anomaly event windows to exclude from training: {len(_anomaly_events)}")
del _events_pd
gc.collect()

all_scores = []

with mlflow.start_run(run_name="IF_Farm_A_parent"):
    mlflow.log_param("n_turbines", len(turbines))
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("contamination_values", CONTAMINATION_VALUES)

    for idx, asset_id in enumerate(turbines):
        # Select ONLY IF columns — avoids pulling unused cols to driver
        turbine_pd = (
            df_spark
            .filter(F.col('asset_id') == asset_id)
            .select(*_IF_COLS)
            .orderBy('time_stamp')
            .toPandas()
        )

        train_mask = turbine_pd['train_test'] == 'train'

        # Exclude rows falling inside anomaly event windows
        turbine_anom = _anomaly_events[_anomaly_events['asset_id'] == int(asset_id)]
        for _, evt in turbine_anom.iterrows():
            train_mask = train_mask & ~(
                (turbine_pd['id'] >= int(evt['event_start_id']))
                & (turbine_pd['id'] <= int(evt['event_end_id']))
            )

        if train_mask.sum() == 0:
            raise ValueError(f"Turbine {asset_id} has no train rows after excluding anomaly windows")

        X_train = (
            turbine_pd.loc[train_mask, feature_cols]
            .fillna(0).to_numpy(dtype=np.float32, copy=False)
        )
        X_all = (
            turbine_pd[feature_cols].fillna(0).to_numpy(dtype=np.float32, copy=False)
        )

        best_model = None
        best_cont  = CONTAMINATION_VALUES[1]
        best_score = np.inf

        with mlflow.start_run(run_name=f"IF_{asset_id}", nested=True):
            for cont in CONTAMINATION_VALUES:
                model = IsolationForest(
                    contamination=cont, random_state=42, n_jobs=1,
                )
                model.fit(X_train)
                val_scores = -model.decision_function(X_train)
                mean_val = float(val_scores.mean())
                mlflow.log_metric(f"mean_score_cont_{cont}", mean_val)
                del val_scores

                if mean_val < best_score:
                    best_score = mean_val
                    best_cont  = cont
                    if best_model is not None:
                        del best_model
                    best_model = model
                else:
                    del model

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

        del turbine_pd, X_train, X_all, scores, best_model, train_mask
        gc.collect()

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

import tempfile, os
_tmpdir = tempfile.mkdtemp()

print("=" * 70)
print("CELL 7 — IF Model Improvement & Diagnostics")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — Feature importance via mean score gap
# Compute feature means for anomaly vs normal event windows using
# Spark aggregation (avoids collecting the full table into driver).
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 1 — Feature importance via mean score gap")
print("=" * 70)

events_spark = spark.table(f"`{CATALOG}`.`wind-farm-a-event-info`")

df_labeled = (
    df_spark
    .join(
        events_spark.select(
            F.col('asset').alias('asset_id'),
            'event_label', 'event_start_id', 'event_end_id',
        ),
        on='asset_id',
        how='inner',
    )
    .filter(
        (F.col('id') >= F.col('event_start_id'))
        & (F.col('id') <= F.col('event_end_id'))
    )
)

anomaly_means_row = (
    df_labeled.filter(F.col('event_label') == 'anomaly')
    .select([F.mean(F.col(c).cast('int')) .alias(c) for c in feature_cols])
    .toPandas()
    .iloc[0]
)
normal_means_row = (
    df_labeled.filter(F.col('event_label') == 'normal')
    .select([F.mean(F.col(c).cast('int')).alias(c) for c in feature_cols])
    .toPandas()
    .iloc[0]
)

sep_records = []
for feat in feature_cols:
    a_mean = float(anomaly_means_row[feat])
    n_mean = float(normal_means_row[feat])
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

feature_sep_df.to_csv(os.path.join(_tmpdir, "if_feature_separation.csv"), index=False)

runs = mlflow.search_runs(
    filter_string="tags.mlflow.runName = 'IF_Farm_A_parent'",
    max_results=1,
)
parent_run_id = runs.iloc[0].run_id
print(f"\nFound parent MLflow run: {parent_run_id}")

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_artifact(os.path.join(_tmpdir, "if_feature_separation.csv"))
print("Logged feature_sep_df as CSV artifact to MLflow.")

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — Retrain using only top discriminating features
# Process one turbine at a time (same pattern as Cell 3) to stay
# within driver memory.
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 2 — Retrain using only top discriminating features")
print("=" * 70)

N_TOP = min(50, len(feature_cols))
top_feature_cols = feature_sep_df.head(N_TOP)['feature'].tolist()
print(f"\nUsing top {len(top_feature_cols)} features for v2 model.")

v2_scores_list = []

_V2_COLS = _META_COLS + top_feature_cols

for turbine_id in sorted(df_scored['asset_id'].unique()):
    turbine_pd = (
        df_spark
        .filter(F.col('asset_id') == int(turbine_id))
        .select(*_V2_COLS)
        .orderBy('time_stamp')
        .toPandas()
    )

    train_mask = (
        (turbine_pd['train_test'] == 'train')
        & (turbine_pd['status_type_id'] == 0)
    )

    # Exclude anomaly event windows from v2 training too
    turbine_anom = _anomaly_events[_anomaly_events['asset_id'] == int(turbine_id)]
    for _, evt in turbine_anom.iterrows():
        train_mask = train_mask & ~(
            (turbine_pd['id'] >= int(evt['event_start_id']))
            & (turbine_pd['id'] <= int(evt['event_end_id']))
        )

    X_train = turbine_pd.loc[train_mask, top_feature_cols].dropna()
    if len(X_train) == 0:
        print(f"  Turbine {turbine_id}: no training data — skipping")
        del turbine_pd
        gc.collect()
        continue

    X_train = X_train.to_numpy(dtype=np.float32, copy=False)
    X_all = turbine_pd[top_feature_cols].fillna(0).to_numpy(dtype=np.float32, copy=False)

    iso_v2 = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_jobs=1,
    )
    iso_v2.fit(X_train)

    scores = -iso_v2.decision_function(X_all)

    threshold_95 = np.percentile(scores, 95)

    v2_scores_list.append(pd.DataFrame({
        'asset_id': turbine_pd['asset_id'].values,
        'id':       turbine_pd['id'].values,
        'if_anomaly_score_v2': scores,
        'if_anomaly_flag_v2':  (scores > threshold_95).astype(int),
    }))

    print(f"  Turbine {turbine_id}: trained on {len(X_train):,} rows, "
          f"scored {len(turbine_pd):,} rows")

    del turbine_pd, X_train, X_all, scores, iso_v2
    gc.collect()

v2_df = pd.concat(v2_scores_list, ignore_index=True)
df_scored = df_scored.merge(v2_df, on=['asset_id', 'id'], how='left')

print(f"\nv2 anomaly flag counts (initial per-turbine 95th percentile):")
print(df_scored['if_anomaly_flag_v2'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════
# Build event-window masks on df_scored for validation steps
# (events DataFrame is already loaded from Cell 4)
# ═══════════════════════════════════════════════════════════════════════

anomaly_mask = pd.Series(False, index=df_scored.index)
normal_mask  = pd.Series(False, index=df_scored.index)
event_desc   = pd.Series(None,  index=df_scored.index, dtype=object)

for _, evt in events.iterrows():
    row_mask = (
        (df_scored['asset_id'] == int(evt['asset_id']))
        & (df_scored['id'] >= int(evt['event_start_id']))
        & (df_scored['id'] <= int(evt['event_end_id']))
    )
    if evt['event_label'] == 'anomaly':
        anomaly_mask |= row_mask
        if pd.notna(evt.get('event_description')):
            event_desc[row_mask] = evt['event_description']
    else:
        normal_mask |= row_mask

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — Re-validate and compare v1 vs v2
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 3 — Re-validate and compare v1 vs v2")
print("=" * 70)

v1_normal_mean  = df_scored.loc[normal_mask,  'if_anomaly_score'].mean()
v1_anomaly_mean = df_scored.loc[anomaly_mask, 'if_anomaly_score'].mean()
v1_sep = v1_anomaly_mean - v1_normal_mean
v1_dir = "OK" if v1_sep > 0 else "WRONG"

v2_normal_mean  = df_scored.loc[normal_mask,  'if_anomaly_score_v2'].mean()
v2_anomaly_mean = df_scored.loc[anomaly_mask, 'if_anomaly_score_v2'].mean()
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
anomaly_with_desc = df_scored.loc[anomaly_mask].copy()
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

event_mask    = anomaly_mask | normal_mask
event_rows    = df_scored[event_mask]
event_anomaly = anomaly_mask[event_mask]
event_normal  = normal_mask[event_mask]

percentiles_to_test = [90, 92, 94, 95, 96, 97, 98, 99]
threshold_results = []

for pct in percentiles_to_test:
    threshold = np.percentile(
        df_scored['if_anomaly_score_v2'].dropna(), pct
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

threshold_df.to_csv(os.path.join(_tmpdir, "if_threshold_comparison.csv"), index=False)
with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_artifact(os.path.join(_tmpdir, "if_threshold_comparison.csv"))
print("Logged threshold comparison CSV to MLflow.")

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5 — Update if_anomaly_flag_v2 using best threshold
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 5 — Update if_anomaly_flag_v2 using best threshold")
print("=" * 70)

best_threshold = np.percentile(
    df_scored['if_anomaly_score_v2'].dropna(), BEST_PERCENTILE
)
df_scored['if_anomaly_flag_v2'] = (
    df_scored['if_anomaly_score_v2'] > best_threshold
).astype(int)

print(f"\nBest threshold (percentile {BEST_PERCENTILE}): {best_threshold:.6f}")
print(f"\nUpdated flag counts:")
print(df_scored['if_anomaly_flag_v2'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT 6 — Log v2 summary metrics to MLflow
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("IMPROVEMENT 6 — Log v2 summary metrics to MLflow")
print("=" * 70)

best_row = threshold_df[threshold_df['percentile'] == BEST_PERCENTILE].iloc[0]
detection_rate_at_best = best_row['detection_rate']
false_alarm_at_best    = best_row['false_alarm_rate']

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("v2_normal_mean_score",  v2_normal_mean)
    mlflow.log_metric("v2_anomaly_mean_score", v2_anomaly_mean)
    mlflow.log_metric("v2_separation_score",   v2_sep)
    mlflow.log_metric("v2_best_percentile",    BEST_PERCENTILE)
    mlflow.log_metric("v2_detection_rate",     detection_rate_at_best)
    mlflow.log_metric("v2_false_alarm_rate",   false_alarm_at_best)
    mlflow.log_param("v2_top_feature_count",   len(top_feature_cols))

print(f"\nLogged to MLflow parent run '{parent_run_id}':")
print(f"  v2_normal_mean_score:  {v2_normal_mean:.6f}")
print(f"  v2_anomaly_mean_score: {v2_anomaly_mean:.6f}")
print(f"  v2_separation_score:   {v2_sep:.6f}")
print(f"  v2_best_percentile:    {BEST_PERCENTILE}")
print(f"  v2_detection_rate:     {detection_rate_at_best:.4f}")
print(f"  v2_false_alarm_rate:   {false_alarm_at_best:.4f}")
print(f"  v2_top_feature_count:  {len(top_feature_cols)}")

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