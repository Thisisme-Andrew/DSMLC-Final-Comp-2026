# Databricks notebook source
# MAGIC %md
# MAGIC # 03c — XGBoost Classifier (Farm A)
# MAGIC Supervised anomaly detection using event-window labels from `wind-farm-a-event-info`.
# MAGIC
# MAGIC **Level 1** — Binary classifier (anomaly vs normal)
# MAGIC **Level 2** — Fault type classifier (on anomaly rows only)
# MAGIC
# MAGIC **Output table contract** — `wind-farm-a-xgb-scores` must include:
# MAGIC `asset_id`, `time_stamp`, `id`, `train_test`, `status_type_id`,
# MAGIC `xgb_anomaly_prob` (float), `xgb_fault_type` (string), `xgb_anomaly_flag` (int 0/1)

# COMMAND ----------

# CELL 1 — Install and restart

%pip install xgboost imbalanced-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELL 2 — Imports and config

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, f1_score,
    precision_score, recall_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from pyspark.sql import functions as F
import pandas as pd, numpy as np
import tempfile, os
import warnings; warnings.filterwarnings('ignore')

CATALOG = "wind-turbine-silver"
_tmpdir = tempfile.mkdtemp()

mlflow.set_experiment(
    "/Users/"
    + spark.sql("SELECT current_user()").first()[0]
    + "/DSMLC-Final-Comp-2026-xgboost-classifier"
)

print(f"Temp directory: {_tmpdir}")

# COMMAND ----------

# CELL 3 — Load and validate

df_spark = spark.table(f"`{CATALOG}`.`wind-farm-a-features`")

required = ['asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id']
missing = [c for c in required if c not in df_spark.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

EXCLUDE_COLS = {
    'asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id',
    'event_id', 'farm', 'event_label', 'event_description',
    'if_anomaly_score', 'if_anomaly_flag',
    'if_anomaly_score_v2', 'if_anomaly_flag_v2',
}
feature_cols = [c for c in df_spark.columns if c not in EXCLUDE_COLS]

if len(feature_cols) == 0:
    raise ValueError("No feature columns — run 02_feature_engineering first")

print(f"Total rows:       {df_spark.count():,}")
print(f"Feature columns:  {len(feature_cols)}")
print(f"Turbines:         {df_spark.select('asset_id').distinct().count()}")

events = spark.table(f"`{CATALOG}`.`wind-farm-a-event-info`").toPandas()
events = events.rename(columns={'asset': 'asset_id'})

print(f"\nEvents loaded: {len(events)}")
print(events['event_label'].value_counts().to_string())

# COMMAND ----------

# CELL 4 — Create row-level labels from event windows
#
# XGBoost is a tabular learner — each row is an independent sample.
# The rolling/delta/deviation features from 02_feature_engineering
# already encode temporal context in every row, so windowed
# flattening is unnecessary and would create a dimension mismatch
# between training (144 × N_features) and per-row scoring (N_features).

# Step 1: Build Spark filter for all event-window rows
filter_conds = []
for _, ev in events.iterrows():
    cond = (
        (F.col('asset_id') == int(ev['asset_id']))
        & (F.col('id') >= int(ev['event_start_id']))
        & (F.col('id') <= int(ev['event_end_id']))
    )
    filter_conds.append(cond)

combined_filter = filter_conds[0]
for cond in filter_conds[1:]:
    combined_filter = combined_filter | cond

# Step 2: Collect ONLY event-window rows (memory-efficient, ~5K rows)
labeled_pd = (
    df_spark
    .filter(combined_filter)
    .select(['asset_id', 'time_stamp', 'id',
             'train_test', 'status_type_id'] + feature_cols)
    .orderBy('asset_id', 'time_stamp')
    .toPandas()
)
print(f"Event-window rows collected: {len(labeled_pd):,}")

# Step 3: Assign labels (anomaly=1 takes precedence over normal=0)
labeled_pd['label'] = -1
labeled_pd['fault_type'] = 'unlabeled'

for _, ev in events.iterrows():
    mask = (
        (labeled_pd['asset_id'] == ev['asset_id'])
        & (labeled_pd['id'] >= ev['event_start_id'])
        & (labeled_pd['id'] <= ev['event_end_id'])
    )
    if ev['event_label'] == 'anomaly':
        labeled_pd.loc[mask, 'label'] = 1
        labeled_pd.loc[mask, 'fault_type'] = ev.get(
            'event_description', 'anomaly')
    else:
        normal_mask = mask & (labeled_pd['label'] == -1)
        labeled_pd.loc[normal_mask, 'label'] = 0
        labeled_pd.loc[normal_mask, 'fault_type'] = 'normal'

labeled_pd = labeled_pd[labeled_pd['label'] >= 0].copy()

print(f"\nLabeled rows: {len(labeled_pd):,}")
print(f"\nLabel distribution:")
print(labeled_pd['label'].value_counts().to_string())
print(f"\nFault type distribution:")
print(labeled_pd['fault_type'].value_counts().to_string())

if labeled_pd['label'].sum() == 0:
    raise ValueError("No anomaly rows found — check event_info")

X_labeled = labeled_pd[feature_cols].fillna(0).values
y_labeled = labeled_pd['label'].values
fault_labels = labeled_pd['fault_type'].values

print(f"\nTraining matrix shape: {X_labeled.shape}")

# COMMAND ----------

# CELL 5 — Class weights and SMOTE check

n_neg = int((y_labeled == 0).sum())
n_pos = int((y_labeled == 1).sum())

if n_neg > 0 and n_pos > 0:
    class_weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_labeled)
    weight_dict = dict(zip([0, 1], class_weights))
    scale_pos_weight_raw = n_neg / n_pos
    # Cap imbalance ratio — very large values push near-1 scores on all rows (high FAR).
    SCALE_POS_CAP = 6.0
    scale_pos_weight = float(min(scale_pos_weight_raw, SCALE_POS_CAP))
else:
    weight_dict = {0: 1.0, 1: 1.0}
    scale_pos_weight_raw = 1.0
    scale_pos_weight = 1.0
    print("WARNING: only one class present in labeled data")

print(f"Class weights: {weight_dict}")
print(f"Normal: {n_neg}, Anomaly: {n_pos}")
print(f"scale_pos_weight (raw / capped): {scale_pos_weight_raw:.2f} / {scale_pos_weight:.2f}")

min_class_count = min(n_neg, n_pos)
USE_SMOTE = min_class_count < 10
print(f"Using SMOTE: {USE_SMOTE} (min class count: {min_class_count})")

# COMMAND ----------

# CELL 6 — Level 1: binary classifier with time-aware CV

# Sort chronologically for TimeSeriesSplit
time_order = labeled_pd['time_stamp'].argsort().values
X_sorted = X_labeled[time_order]
y_sorted = y_labeled[time_order]
fault_sorted = fault_labels[time_order]

# Hold out the last slice of time for threshold calibration (not used for training).
# Tuning thresholds on training scores inflated FAR / broke precision.
HOLDOUT_FRAC = 0.20
MIN_RECALL_CAL = 0.75
n_lab = len(X_sorted)
holdout_start = max(50, int(n_lab * (1.0 - HOLDOUT_FRAC)))
if holdout_start >= n_lab - 20:
    holdout_start = int(n_lab * 0.85)
X_fit = X_sorted[:holdout_start]
y_fit = y_sorted[:holdout_start]
fault_fit = fault_sorted[:holdout_start]
X_cal = X_sorted[holdout_start:]
y_cal = y_sorted[holdout_start:]
print(f"\nTemporal split: fit rows={len(X_fit):,}  calibration rows={len(X_cal):,}")
print(f"  Fit label balance: normal={(y_fit==0).sum():,}  anomaly={(y_fit==1).sum():,}")
print(f"  Cal label balance: normal={(y_cal==0).sum():,}  anomaly={(y_cal==1).sum():,}")

n_splits_cv = min(5, max(2, len(X_fit) // 200))
tscv = TimeSeriesSplit(n_splits=n_splits_cv, gap=20)

# Stronger regularization + slightly shallower trees → fewer spurious positives.
def _xgb_l1_params():
    return dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=4,
        gamma=0.15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.5,
        reg_alpha=0.3,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0,
    )

fold_metrics = []

with mlflow.start_run(run_name="xgb_level1_binary") as run:
    xgb_run_id = run.info.run_id

    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_labeled_rows", len(y_sorted))
    mlflow.log_param("n_anomaly", int(y_sorted.sum()))
    mlflow.log_param("n_normal", int((y_sorted == 0).sum()))
    mlflow.log_param("use_smote", USE_SMOTE)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
    mlflow.log_param("holdout_frac", HOLDOUT_FRAC)
    mlflow.log_param("min_recall_cal", MIN_RECALL_CAL)
    mlflow.log_param("fit_rows", len(X_fit))
    mlflow.log_param("cal_rows", len(X_cal))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_fit)):
        X_tr, X_val = X_fit[train_idx], X_fit[val_idx]
        y_tr, y_val = y_fit[train_idx], y_fit[val_idx]

        n_classes_tr = len(np.unique(y_tr))

        if n_classes_tr < 2:
            print(f"Fold {fold}: SKIPPED — training data has only "
                  f"class {np.unique(y_tr)[0]} ({len(y_tr)} rows)")
            fold_metrics.append({
                'fold': fold, 'f1': 0.0,
                'precision': 0.0, 'recall': 0.0,
                'val_size': len(y_val),
                'val_pos': int(y_val.sum()),
            })
            continue

        if (USE_SMOTE and y_tr.sum() >= 2
                and (y_tr == 0).sum() >= 2):
            k = min(3, int(y_tr.sum()) - 1)
            sm = SMOTE(k_neighbors=max(1, k), random_state=42)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        model = XGBClassifier(**_xgb_l1_params())
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        f1  = f1_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        rec  = recall_score(y_val, preds, zero_division=0)

        fold_metrics.append({
            'fold': fold, 'f1': f1,
            'precision': prec, 'recall': rec,
            'val_size': len(y_val),
            'val_pos': int(y_val.sum()),
        })

        mlflow.log_metric(f"fold_{fold}_f1", f1)
        mlflow.log_metric(f"fold_{fold}_precision", prec)
        mlflow.log_metric(f"fold_{fold}_recall", rec)
        print(f"Fold {fold}: F1={f1:.3f}  Prec={prec:.3f}  "
              f"Rec={rec:.3f}  (val={len(y_val)}, pos={y_val.sum()})")

    metrics_df = pd.DataFrame(fold_metrics)
    mean_f1 = metrics_df['f1'].mean()
    mlflow.log_metric("mean_cv_f1", mean_f1)
    print(f"\nMean CV F1: {mean_f1:.3f}")

    # Final model: fit ONLY on non-holdout rows (same distribution as deployment).
    if (USE_SMOTE and y_fit.sum() >= 2
            and (y_fit == 0).sum() >= 2):
        k = min(3, int(y_fit.sum()) - 1)
        sm = SMOTE(k_neighbors=max(1, k), random_state=42)
        X_final, y_final = sm.fit_resample(X_fit, y_fit)
    else:
        X_final, y_final = X_fit, y_fit

    xgb_l1 = XGBClassifier(**_xgb_l1_params())
    xgb_l1.fit(X_final, y_final)

    if len(y_cal) > 0 and len(np.unique(y_cal)) > 1:
        cal_ap = average_precision_score(y_cal, xgb_l1.predict_proba(X_cal)[:, 1])
        mlflow.log_metric("cal_pr_auc", cal_ap)
        print(f"Calibration PR-AUC (holdout): {cal_ap:.4f}")

    mlflow.xgboost.log_model(xgb_l1, "xgb_level1")

    # Feature importance
    importance = pd.Series(
        xgb_l1.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    imp_path = os.path.join(_tmpdir, "xgb_feature_importance.csv")
    importance.to_csv(imp_path)
    mlflow.log_artifact(imp_path)

    cv_path = os.path.join(_tmpdir, "xgb_cv_metrics.csv")
    metrics_df.to_csv(cv_path, index=False)
    mlflow.log_artifact(cv_path)

    print(f"\nTop 20 features by importance:")
    print(importance.head(20).to_string())

# COMMAND ----------

# CELL 7 — Level 2: fault type classifier (fit split only — no leakage into cal)

anomaly_mask = y_fit == 1
X_anom = X_fit[anomaly_mask]
f_anom = fault_fit[anomaly_mask]

fault_types = sorted(np.unique(f_anom))
fault_to_int = {f: i for i, f in enumerate(fault_types)}
int_to_fault = {i: f for f, i in fault_to_int.items()}
y_fault = np.array([fault_to_int[f] for f in f_anom])

print(f"Anomaly rows for Level 2: {len(X_anom)}")
print(f"Fault types ({len(fault_types)}): {fault_types}")
print(f"\nFault type distribution:")
print(pd.Series(f_anom).value_counts().to_string())

HAS_L2 = len(X_anom) >= 5 and len(fault_types) > 1

if not HAS_L2:
    print("\nWARNING: too few anomaly samples or only 1 fault type "
          "— skipping Level 2")
    xgb_l2 = None
else:
    with mlflow.start_run(run_name="xgb_level2_faulttype"):
        xgb_l2 = XGBClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        xgb_l2.fit(X_anom, y_fault)

        l2_preds = xgb_l2.predict(X_anom)
        print("\nLevel 2 classification report (training set):")
        print(classification_report(
            y_fault, l2_preds,
            target_names=fault_types,
            zero_division=0,
        ))

        mlflow.xgboost.log_model(xgb_l2, "xgb_level2")
        mlflow.log_param("fault_types", str(fault_types))
        mlflow.log_param("n_fault_classes", len(fault_types))

# COMMAND ----------

# CELL 8 — Threshold tuning and full-dataframe scoring

# ── Step 1: Threshold on TIME HOLDOUT (reduces false alarms vs in-sample tuning) ──
def _fbeta(prec, rec, beta=0.5):
    """F-beta with beta<1 weights precision higher than recall."""
    b2 = beta * beta
    denom = b2 * prec + rec
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * prec * rec / denom


if len(X_cal) < 40 or len(np.unique(y_cal)) < 2:
    print("WARNING: calibration set small or single-class — "
          "using full labeled set for threshold search.")
    X_thr, y_thr = X_sorted, y_sorted
else:
    X_thr, y_thr = X_cal, y_cal

cal_probs = xgb_l1.predict_proba(X_thr)[:, 1]

threshold_grid = np.unique(np.clip(
    np.concatenate([
        np.linspace(0.35, 0.95, 80),
        np.linspace(0.95, 0.999, 40),
    ]),
    0.0, 1.0,
))

threshold_results = []
for thr in threshold_grid:
    preds = (cal_probs > thr).astype(int)
    _tp = int(((preds == 1) & (y_thr == 1)).sum())
    _fn = int(((preds == 0) & (y_thr == 1)).sum())
    _fp = int(((preds == 1) & (y_thr == 0)).sum())
    _tn = int(((preds == 0) & (y_thr == 0)).sum())
    _det  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
    _far  = _fp / (_fp + _tn) if (_fp + _tn) > 0 else 0
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
    _f1   = 2 * _prec * _det / (_prec + _det) if (_prec + _det) > 0 else 0
    _f05  = _fbeta(_prec, _det, beta=0.5)
    threshold_results.append({
        'threshold': thr, 'TP': _tp, 'FP': _fp, 'FN': _fn, 'TN': _tn,
        'detection_rate': round(_det, 4),
        'false_alarm_rate': round(_far, 4),
        'precision': round(_prec, 4),
        'f1': round(_f1, 4),
        'f05': round(_f05, 4),
    })

thresh_df = pd.DataFrame(threshold_results)
print("Threshold comparison on calibration (holdout) event-window rows:\n")
disp_cols = ['threshold', 'detection_rate', 'false_alarm_rate',
               'precision', 'f1', 'f05']
print(thresh_df[disp_cols].to_string(index=False))

# Prefer precision: among thresholds meeting min recall on cal, take highest precision.
meet = thresh_df[thresh_df['detection_rate'] >= MIN_RECALL_CAL]
if len(meet) > 0:
    best_idx = meet['precision'].idxmax()
    TUNED_THRESHOLD = float(meet.loc[best_idx, 'threshold'])
    sel_reason = f"max precision subject to recall>={MIN_RECALL_CAL}"
else:
    best_idx = thresh_df['f05'].idxmax()
    TUNED_THRESHOLD = float(thresh_df.loc[best_idx, 'threshold'])
    sel_reason = "no threshold met min recall — max F0.5 on calibration"

print(f"\nSelected threshold: {TUNED_THRESHOLD:.4f}  ({sel_reason})")

thresh_csv = os.path.join(_tmpdir, "xgb_threshold_comparison.csv")
thresh_df.to_csv(thresh_csv, index=False)
with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_artifact(thresh_csv)
    mlflow.log_param("tuned_threshold", TUNED_THRESHOLD)
    mlflow.log_param("threshold_strategy", sel_reason)

# ── Step 2: Score full dataframe per-turbine ──────────────────────

all_turbines = sorted(
    [r.asset_id for r in df_spark.select('asset_id').distinct().collect()]
)

scored_parts = []

for asset_id in all_turbines:
    print(f"Scoring turbine {asset_id} ...")

    turbine_pd = (
        df_spark
        .filter(F.col('asset_id') == int(asset_id))
        .select(['asset_id', 'time_stamp', 'id',
                 'train_test', 'status_type_id'] + feature_cols)
        .orderBy('time_stamp')
        .toPandas()
    )

    X_turbine = turbine_pd[feature_cols].fillna(0).values

    probs = xgb_l1.predict_proba(X_turbine)[:, 1]
    flags = (probs > TUNED_THRESHOLD).astype(int)

    turbine_scored = turbine_pd[
        ['asset_id', 'time_stamp', 'id',
         'train_test', 'status_type_id']
    ].copy()
    turbine_scored['xgb_anomaly_prob'] = probs
    turbine_scored['xgb_anomaly_flag'] = flags

    if xgb_l2 is not None and flags.sum() > 0:
        fault_preds = np.full(len(probs), 'normal', dtype=object)
        anom_idx = np.where(flags == 1)[0]
        ft_ints = xgb_l2.predict(X_turbine[anom_idx])
        fault_preds[anom_idx] = [
            int_to_fault[int(i)] for i in ft_ints
        ]
        turbine_scored['xgb_fault_type'] = fault_preds
    elif flags.sum() > 0:
        turbine_scored['xgb_fault_type'] = np.where(
            flags == 1, fault_types[0] if fault_types else 'unknown',
            'normal',
        )
    else:
        turbine_scored['xgb_fault_type'] = 'normal'

    scored_parts.append(turbine_scored)

    n_flagged = int(flags.sum())
    print(f"  {asset_id}: {len(turbine_pd):,} rows, "
          f"{n_flagged:,} flagged ({n_flagged/len(turbine_pd)*100:.1f}%)")

    del X_turbine, turbine_pd

df_scored = pd.concat(scored_parts, ignore_index=True)
del scored_parts

# ── Step 3: Continuous risk tier ──────────────────────────────────
# xgb_anomaly_prob is the continuous risk metric (0–1).
# xgb_risk_tier bins it into actionable categories.
df_scored['xgb_risk_tier'] = pd.cut(
    df_scored['xgb_anomaly_prob'],
    bins=[-0.01, 0.3, 0.6, 0.9, 1.01],
    labels=['low', 'medium', 'high', 'critical'],
).astype(str)

print(f"\nTotal scored rows:  {len(df_scored):,}")
print(f"Total flagged (threshold={TUNED_THRESHOLD}):  "
      f"{df_scored['xgb_anomaly_flag'].sum():,}")
print(f"\nRisk tier distribution:")
print(df_scored['xgb_risk_tier'].value_counts().to_string())
print(f"\nFault type distribution (flagged rows):")
print(df_scored[df_scored['xgb_anomaly_flag'] == 1]
    ['xgb_fault_type'].value_counts().to_string())

# COMMAND ----------

# CELL 9 — Validation against event windows

df_val = df_scored.merge(
    events[['asset_id', 'event_id', 'event_label',
            'event_start_id', 'event_end_id', 'event_description']],
    on='asset_id', how='inner',
)
df_val['in_event'] = (
    (df_val['id'] >= df_val['event_start_id'])
    & (df_val['id'] <= df_val['event_end_id'])
)
df_val = df_val[df_val['in_event']].copy()

anomaly_val = df_val[df_val['event_label'] == 'anomaly']
normal_val  = df_val[df_val['event_label'] == 'normal']

print("XGBoost anomaly probability by event label:")
print(df_val.groupby('event_label')['xgb_anomaly_prob'].agg(
    ['mean', 'std', 'count']
))

separation = (anomaly_val['xgb_anomaly_prob'].mean()
              - normal_val['xgb_anomaly_prob'].mean())
direction = "OK" if separation > 0 else "WRONG"
print(f"\nSeparation (anomaly - normal): {separation:.4f}  →  {direction}")

# Compare default 0.5 vs tuned threshold on event windows
for label, thr in [("default (0.5)", 0.5),
                   (f"tuned ({TUNED_THRESHOLD})", TUNED_THRESHOLD)]:
    a_flag = (anomaly_val['xgb_anomaly_prob'] > thr).astype(int)
    n_flag = (normal_val['xgb_anomaly_prob'] > thr).astype(int)
    _tp = int(a_flag.sum())
    _fn = int((a_flag == 0).sum())
    _fp = int(n_flag.sum())
    _tn = int((n_flag == 0).sum())
    _det  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
    _far  = _fp / (_fp + _tn) if (_fp + _tn) > 0 else 0
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
    print(f"\n── Threshold: {label} ──")
    print(f"  Detection rate (recall):  {_det:.4f}")
    print(f"  False alarm rate:         {_far:.4f}")
    print(f"  Precision:                {_prec:.4f}")
    print(f"  TP={_tp}, FN={_fn}, FP={_fp}, TN={_tn}")

# Metrics at the tuned threshold (used for MLflow)
tp = int((anomaly_val['xgb_anomaly_flag'] == 1).sum())
fn = int((anomaly_val['xgb_anomaly_flag'] == 0).sum())
fp = int((normal_val['xgb_anomaly_flag'] == 1).sum())
tn = int((normal_val['xgb_anomaly_flag'] == 0).sum())

det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
fa_rate  = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# Risk tier breakdown in event windows
print(f"\n── Risk tier breakdown in event windows ──")
print("Anomaly events:")
print(pd.cut(
    anomaly_val['xgb_anomaly_prob'],
    bins=[-0.01, 0.3, 0.6, 0.9, 1.01],
    labels=['low', 'medium', 'high', 'critical'],
).value_counts().to_string())
print("\nNormal events:")
print(pd.cut(
    normal_val['xgb_anomaly_prob'],
    bins=[-0.01, 0.3, 0.6, 0.9, 1.01],
    labels=['low', 'medium', 'high', 'critical'],
).value_counts().to_string())

# Per-fault-type detection
if anomaly_val['xgb_anomaly_flag'].sum() > 0:
    print("\nPer-fault-type detection rate:")
    for ft in sorted(anomaly_val['event_description'].unique()):
        ft_rows = anomaly_val[anomaly_val['event_description'] == ft]
        ft_det = ft_rows['xgb_anomaly_flag'].mean()
        print(f"  {ft}: {ft_det:.3f} ({ft_rows['xgb_anomaly_flag'].sum()}/{len(ft_rows)})")

with mlflow.start_run(run_id=xgb_run_id):
    mlflow.log_metric("detection_rate", det_rate)
    mlflow.log_metric("false_alarm_rate", fa_rate)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("separation", separation)

print("\nValidation metrics logged to MLflow.")

# COMMAND ----------

# CELL 10 — Output schema validation

required_out = [
    'asset_id', 'time_stamp', 'id',
    'train_test', 'status_type_id',
    'xgb_anomaly_prob', 'xgb_anomaly_flag', 'xgb_fault_type',
    'xgb_risk_tier',
]
for col in required_out:
    if col not in df_scored.columns:
        raise ValueError(f"Output missing required column: {col}")

dupes = df_scored.duplicated(
    subset=['asset_id', 'time_stamp', 'id']).sum()
if dupes > 0:
    raise ValueError(f"Duplicate keys in output: {dupes} rows")

print("Output schema validated. No duplicate keys.")
print(df_scored[required_out].dtypes)
print(f"\nTotal rows: {len(df_scored):,}")
print(f"Null counts:\n{df_scored[required_out].isnull().sum()}")
print(f"\nFlag distribution (threshold={TUNED_THRESHOLD}):")
print(df_scored['xgb_anomaly_flag'].value_counts().to_string())
print(f"\nRisk tier distribution:")
print(df_scored['xgb_risk_tier'].value_counts().to_string())

# COMMAND ----------

# CELL 11 — Save to Delta

(
    spark.createDataFrame(df_scored[required_out])
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`wind-farm-a-xgb-scores`")
)

saved = spark.table(f"`{CATALOG}`.`wind-farm-a-xgb-scores`")
print(f"Saved rows: {saved.count():,}")
print(f"Columns:    {saved.columns}")

assert 'xgb_anomaly_prob'  in saved.columns
assert 'xgb_anomaly_flag'  in saved.columns
assert 'xgb_fault_type'    in saved.columns
print("\nAll required columns confirmed in Delta table.")

saved.groupBy('xgb_anomaly_flag').count().show()
saved.groupBy('xgb_fault_type').count().show()
saved.groupBy('xgb_risk_tier').count().show()