# Databricks notebook source
# MAGIC %md
# MAGIC # 03b — LSTM Autoencoder (Farm A)
# MAGIC Sequence-based anomaly detection using reconstruction error.
# MAGIC
# MAGIC **Stretch goal** — complementary to the Isolation Forest in 03a.
# MAGIC
# MAGIC **Output table contract** — `wind-farm-a-lstm-scores` must include:
# MAGIC `asset_id`, `time_stamp`, `id`, `train_test`, `status_type_id`,
# MAGIC `lstm_recon_error` (float), `lstm_anomaly_flag` (int 0/1),
# MAGIC `lstm_anomaly_flag_v2` (int 0/1)

# COMMAND ----------

# CELL 1 — Install and restart

%pip install tensorflow mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# CELL 2 — Imports and config

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector,
    TimeDistributed, Dense, Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
)
from pyspark.sql import functions as F
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import tempfile, os, pickle
import warnings; warnings.filterwarnings('ignore')

CATALOG      = "wind-turbine-silver"
WINDOW_SIZE  = 72     # 12 hours at 10-min intervals (halved for CPU speed)
WINDOW_STEP  = 24     # 4-hour stride for training (reduces memory 4x)
BATCH_SIZE   = 64
MAX_EPOCHS   = 30
MAX_FEATURES = 30     # cap features before LSTM (lighter for CPU)

_tmpdir = tempfile.mkdtemp()

mlflow.set_experiment(
    "/Users/"
    + spark.sql("SELECT current_user()").first()[0]
    + "/DSMLC-Final-Comp-2026-lstm-autoencoder"
)

print(f"TensorFlow version: {tf.__version__}")
print(f"Temp directory:     {_tmpdir}")

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
all_feature_cols = [c for c in df_spark.columns if c not in EXCLUDE_COLS]

if len(all_feature_cols) == 0:
    raise ValueError("No feature columns found — check 02_feature_engineering ran first")

print(f"Total available features: {len(all_feature_cols)}")
print(f"Will reduce to top {MAX_FEATURES} by variance")
print(f"Total rows: {df_spark.count():,}")
print(f"Turbines:   {df_spark.select('asset_id').distinct().count()}")

df_spark.groupBy('train_test', 'status_type_id').count().orderBy(
    'train_test', 'status_type_id'
).show()

normal_train_count = df_spark.filter(
    (F.col('status_type_id') == 0) & (F.col('train_test') == 'train')
).count()
if normal_train_count == 0:
    raise ValueError("No normal train rows found — cannot train autoencoder")
print(f"Normal train rows available: {normal_train_count:,}")

# COMMAND ----------

# CELL 4 — Feature reduction (top 50 by variance on train rows)

from pyspark.sql.functions import variance as spark_var

normal_train_spark = df_spark.filter(
    (F.col('status_type_id') == 0) & (F.col('train_test') == 'train')
)

var_exprs = [spark_var(F.col(c)).alias(c) for c in all_feature_cols]
var_row = normal_train_spark.select(var_exprs).first()

variance_series = pd.Series(
    {c: (var_row[c] if var_row[c] is not None else 0.0) for c in all_feature_cols}
).sort_values(ascending=False)

feature_cols = variance_series.head(MAX_FEATURES).index.tolist()

print(f"Selected top {len(feature_cols)} features by variance:")
print(variance_series.head(MAX_FEATURES).to_string())

if len(feature_cols) == 0:
    raise ValueError("Feature reduction produced 0 columns — check input data")

# COMMAND ----------

# CELL 5 — Build the LSTM Autoencoder

def build_autoencoder(window_size, n_features, latent_dim=16, dropout_rate=0.1):
    inputs = Input(shape=(window_size, n_features), name="encoder_input")

    encoded = LSTM(32, return_sequences=False,
                   name="encoder_lstm")(inputs)
    encoded = Dropout(dropout_rate)(encoded)

    repeated = RepeatVector(window_size, name="repeat")(encoded)

    x = LSTM(32, return_sequences=True,
             name="decoder_lstm")(repeated)
    x = Dropout(dropout_rate)(x)
    outputs = TimeDistributed(Dense(n_features), name="output")(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer='adam', loss='mse')
    return model

autoencoder = build_autoencoder(
    window_size=WINDOW_SIZE,
    n_features=len(feature_cols),
)
autoencoder.summary()
print(f"\nInput shape:  (batch, {WINDOW_SIZE}, {len(feature_cols)})")
print(f"Output shape: (batch, {WINDOW_SIZE}, {len(feature_cols)})")

# COMMAND ----------

# CELL 6 — Prepare training windows (normal rows only)

def make_windows(df_turbine, feature_cols, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    df_sorted = df_turbine.sort_values('time_stamp').reset_index(drop=True)
    vals = df_sorted[feature_cols].fillna(0).values
    ts   = df_sorted['time_stamp'].values
    ids  = df_sorted['id'].values

    X, timestamps, id_list = [], [], []
    for i in range(0, len(vals) - window_size + 1, step):
        X.append(vals[i : i + window_size])
        timestamps.append(ts[i + window_size - 1])
        id_list.append(ids[i + window_size - 1])

    return np.array(X, dtype=np.float32), timestamps, id_list

normal_train_pd = (
    df_spark
    .filter(
        (F.col('status_type_id') == 0) & (F.col('train_test') == 'train')
    )
    .select(['asset_id', 'time_stamp', 'id'] + feature_cols)
    .orderBy('asset_id', 'time_stamp')
    .toPandas()
)
print(f"Normal train rows collected: {len(normal_train_pd):,}")

scaler = StandardScaler()
scaler.fit(normal_train_pd[feature_cols].fillna(0))
print("StandardScaler fitted on normal train rows only.")

all_train_windows = []
turbines = sorted(normal_train_pd['asset_id'].unique())

for asset_id in turbines:
    turbine_df = normal_train_pd[normal_train_pd['asset_id'] == asset_id].copy()
    turbine_df[feature_cols] = scaler.transform(turbine_df[feature_cols].fillna(0))

    X_w, _, _ = make_windows(turbine_df, feature_cols)
    if len(X_w) == 0:
        print(f"  WARNING: Turbine {asset_id} produced 0 windows — skipping")
        continue

    all_train_windows.append(X_w)
    print(f"  Turbine {asset_id}: {len(X_w):,} training windows")

if len(all_train_windows) == 0:
    raise ValueError("No training windows created — check window size vs row count")

X_train = np.concatenate(all_train_windows, axis=0)
del all_train_windows

print(f"\nTotal training windows: {X_train.shape[0]:,}")
print(f"Window shape:           {X_train.shape}")

mem_gb = X_train.nbytes / 1e9
print(f"Training array size:    {mem_gb:.2f} GB")
if mem_gb > 4.0:
    print("WARNING: training array > 4 GB — consider increasing WINDOW_STEP or reducing MAX_FEATURES")

# COMMAND ----------

# CELL 7 — Train the autoencoder with MLflow tracking

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1,
    ),
]

with mlflow.start_run(run_name="LSTM_AE_Farm_A") as run:
    lstm_run_id = run.info.run_id

    mlflow.log_param("window_size",     WINDOW_SIZE)
    mlflow.log_param("window_step",     WINDOW_STEP)
    mlflow.log_param("n_features",      len(feature_cols))
    mlflow.log_param("latent_dim",      32)
    mlflow.log_param("dropout_rate",    0.1)
    mlflow.log_param("batch_size",      BATCH_SIZE)
    mlflow.log_param("max_epochs",      MAX_EPOCHS)
    mlflow.log_param("n_train_windows", X_train.shape[0])

    history = autoencoder.fit(
        X_train, X_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    final_train_loss = history.history['loss'][-1]
    final_val_loss   = history.history['val_loss'][-1]
    epochs_run       = len(history.history['loss'])

    mlflow.log_metric("final_train_loss", final_train_loss)
    mlflow.log_metric("final_val_loss",   final_val_loss)
    mlflow.log_metric("epochs_run",       epochs_run)

    mlflow.tensorflow.log_model(autoencoder, "lstm_autoencoder")

    scaler_path = os.path.join(_tmpdir, "lstm_scaler_a.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact(scaler_path)

print(f"\nTraining complete.  Run ID: {lstm_run_id}")
print(f"  Epochs run:       {epochs_run}")
print(f"  Final train loss: {final_train_loss:.6f}")
print(f"  Final val loss:   {final_val_loss:.6f}")

del X_train

# COMMAND ----------

# CELL 8 — Score ALL rows (train + test, all status types)
# Stream windows in batches per turbine to avoid OOM.

SCORE_BATCH = 256

all_turbines = sorted(
    [r.asset_id for r in df_spark.select('asset_id').distinct().collect()]
)

score_records = []

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

    turbine_pd[feature_cols] = scaler.transform(
        turbine_pd[feature_cols].fillna(0)
    )

    vals = turbine_pd[feature_cols].values.astype(np.float32)
    ts   = turbine_pd['time_stamp'].values
    ids  = turbine_pd['id'].values
    n_windows = len(vals) - WINDOW_SIZE + 1

    if n_windows <= 0:
        print(f"  WARNING: {asset_id} — too few rows "
              f"({len(turbine_pd)}) for window size {WINDOW_SIZE}. Skipping.")
        continue

    turbine_errors = []

    for start in range(0, n_windows, SCORE_BATCH):
        end = min(start + SCORE_BATCH, n_windows)
        batch = np.array(
            [vals[i : i + WINDOW_SIZE] for i in range(start, end)],
            dtype=np.float32,
        )
        recon = autoencoder.predict(batch, verbose=0)
        mse = np.mean(np.square(batch - recon), axis=(1, 2))
        turbine_errors.extend(mse.tolist())
        del batch, recon

    for i, err in enumerate(turbine_errors):
        row_idx = i + WINDOW_SIZE - 1
        score_records.append({
            'asset_id':         int(asset_id),
            'time_stamp':       ts[row_idx],
            'id':               int(ids[row_idx]),
            'lstm_recon_error': float(err),
        })

    mean_err = np.mean(turbine_errors)
    print(f"  {asset_id}: {len(turbine_errors):,} windows scored, "
          f"mean recon error = {mean_err:.6f}")
    del vals, turbine_errors

scores_df = pd.DataFrame(score_records)
del score_records
print(f"\nTotal scored records: {len(scores_df):,}")

full_meta = (
    df_spark
    .select(['asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id'])
    .toPandas()
)

df_scored_lstm = full_meta.merge(
    scores_df, on=['asset_id', 'time_stamp', 'id'], how='left'
)
df_scored_lstm['lstm_recon_error'] = df_scored_lstm['lstm_recon_error'].fillna(0.0)
del full_meta, scores_df

print(f"Final scored rows:      {len(df_scored_lstm):,}")
print(f"Rows with score=0 (no window): "
      f"{(df_scored_lstm['lstm_recon_error'] == 0).sum():,}")

# COMMAND ----------

# CELL 9 — Threshold tuning and anomaly flags

p95 = np.percentile(df_scored_lstm['lstm_recon_error'], 95)
df_scored_lstm['lstm_anomaly_flag'] = (
    df_scored_lstm['lstm_recon_error'] > p95
).astype(int)

print(f"v1 threshold (p95): {p95:.6f}")
print(f"v1 flag counts:\n{df_scored_lstm['lstm_anomaly_flag'].value_counts().to_string()}\n")

events = spark.table(f"`{CATALOG}`.`wind-farm-a-event-info`").toPandas()
events = events.rename(columns={'asset': 'asset_id'})

df_eval = df_scored_lstm.merge(
    events[['asset_id', 'event_id', 'event_label',
            'event_start_id', 'event_end_id', 'event_description']],
    on='asset_id', how='inner',
)
df_eval['in_event'] = (
    (df_eval['id'] >= df_eval['event_start_id'])
    & (df_eval['id'] <= df_eval['event_end_id'])
)
df_eval = df_eval[df_eval['in_event']].copy()

print("LSTM reconstruction error by event label:")
print(df_eval.groupby('event_label')['lstm_recon_error'].agg(
    ['mean', 'std', 'count']
))
print()

anomaly_eval = df_eval[df_eval['event_label'] == 'anomaly']
normal_eval  = df_eval[df_eval['event_label'] == 'normal']

separation = anomaly_eval['lstm_recon_error'].mean() - normal_eval['lstm_recon_error'].mean()
direction  = "OK" if separation > 0 else "WRONG"
print(f"Separation (anomaly - normal): {separation:.6f}  →  {direction}\n")

threshold_rows = []
for pct in [90, 92, 94, 95, 96, 97, 98, 99]:
    thresh = np.percentile(df_scored_lstm['lstm_recon_error'], pct)

    tp = int((anomaly_eval['lstm_recon_error'] > thresh).sum())
    fn = int((anomaly_eval['lstm_recon_error'] <= thresh).sum())
    fp = int((normal_eval['lstm_recon_error']  > thresh).sum())
    tn = int((normal_eval['lstm_recon_error']  <= thresh).sum())

    det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    fa_rate  = fp / (fp + tn) if (fp + tn) > 0 else 0

    threshold_rows.append({
        'percentile':       pct,
        'threshold':        round(float(thresh), 6),
        'detection_rate':   round(det_rate, 4),
        'false_alarm_rate': round(fa_rate, 4),
    })

thresh_df = pd.DataFrame(threshold_rows)
print("Threshold comparison:")
print(thresh_df.to_string(index=False))

valid = thresh_df[thresh_df['false_alarm_rate'] < 0.15]
if len(valid) == 0:
    print("\nWARNING: no threshold achieves FA < 0.15 — using p95 as fallback")
    BEST_PCT = 95
else:
    BEST_PCT = int(valid.loc[valid['detection_rate'].idxmax(), 'percentile'])
print(f"\nRecommended threshold percentile: {BEST_PCT}")

best_thresh = np.percentile(df_scored_lstm['lstm_recon_error'], BEST_PCT)
df_scored_lstm['lstm_anomaly_flag_v2'] = (
    df_scored_lstm['lstm_recon_error'] > best_thresh
).astype(int)

print(f"\nv2 flag counts (p{BEST_PCT}):")
print(df_scored_lstm['lstm_anomaly_flag_v2'].value_counts().to_string())

best_row = thresh_df[thresh_df['percentile'] == BEST_PCT].iloc[0]

with mlflow.start_run(run_id=lstm_run_id):
    mlflow.log_metric("lstm_separation",       separation)
    mlflow.log_metric("lstm_best_percentile",  BEST_PCT)
    mlflow.log_metric("lstm_detection_rate",   float(best_row['detection_rate']))
    mlflow.log_metric("lstm_false_alarm_rate", float(best_row['false_alarm_rate']))

    thresh_csv = os.path.join(_tmpdir, "lstm_threshold_comparison.csv")
    thresh_df.to_csv(thresh_csv, index=False)
    mlflow.log_artifact(thresh_csv)

print("\nThreshold metrics logged to MLflow.")

# COMMAND ----------

# CELL 10 — Validation plot (case study turbine)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

top_turbine = (
    df_eval[df_eval['event_label'] == 'anomaly']
    .groupby('asset_id').size()
    .idxmax()
)
print(f"Case study turbine: {top_turbine}")

plot_df = df_scored_lstm[
    df_scored_lstm['asset_id'] == top_turbine
].sort_values('time_stamp').copy()

turbine_events = events[events['asset_id'] == top_turbine]

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(
    plot_df['time_stamp'], plot_df['lstm_recon_error'],
    color='steelblue', linewidth=0.5, label='Reconstruction error',
)
ax.axhline(y=best_thresh, color='red', linewidth=1.5, linestyle='--',
           label=f'Threshold (p{BEST_PCT})')
ax.axhline(y=p95, color='orange', linewidth=1, linestyle=':',
           label='p95 (v1 threshold)')

labels_drawn = set()
for _, ev in turbine_events.iterrows():
    ev_rows = plot_df[
        (plot_df['id'] >= ev['event_start_id'])
        & (plot_df['id'] <= ev['event_end_id'])
    ]
    if len(ev_rows) == 0:
        continue
    color = 'red' if ev['event_label'] == 'anomaly' else 'green'
    lbl = f"{ev['event_label']} window" if ev['event_label'] not in labels_drawn else None
    labels_drawn.add(ev['event_label'])
    ax.axvspan(
        ev_rows['time_stamp'].min(), ev_rows['time_stamp'].max(),
        color=color, alpha=0.15, label=lbl,
    )

ax.set_title(f"LSTM Reconstruction Error — Turbine {top_turbine}")
ax.set_xlabel("Time")
ax.set_ylabel("Reconstruction MSE")
ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()

plot_path = os.path.join(_tmpdir, f"lstm_recon_{top_turbine}.png")
plt.savefig(plot_path, dpi=150)
plt.show()

with mlflow.start_run(run_id=lstm_run_id):
    mlflow.log_artifact(plot_path)
print("Plot saved and logged to MLflow.")

# COMMAND ----------

# CELL 11 — Output schema validation

required_out = [
    'asset_id', 'time_stamp', 'id',
    'train_test', 'status_type_id',
    'lstm_recon_error',
    'lstm_anomaly_flag',
    'lstm_anomaly_flag_v2',
]
for col in required_out:
    if col not in df_scored_lstm.columns:
        raise ValueError(f"Output missing required column: {col}")

dupes = df_scored_lstm.duplicated(subset=['asset_id', 'time_stamp', 'id']).sum()
if dupes > 0:
    raise ValueError(f"Duplicate keys in output: {dupes} rows")

print("Output schema validated. No duplicate keys.")
print(df_scored_lstm[required_out].dtypes)
print(f"\nTotal rows:  {len(df_scored_lstm):,}")
print(f"Null counts:\n{df_scored_lstm[required_out].isnull().sum()}")
print(f"\nFlag distribution (v1):")
print(df_scored_lstm['lstm_anomaly_flag'].value_counts().to_string())
print(f"\nFlag distribution (v2, p{BEST_PCT}):")
print(df_scored_lstm['lstm_anomaly_flag_v2'].value_counts().to_string())

# COMMAND ----------

# CELL 12 — Save to Delta

(
    spark.createDataFrame(df_scored_lstm[required_out])
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"`{CATALOG}`.`wind-farm-a-lstm-scores`")
)

saved = spark.table(f"`{CATALOG}`.`wind-farm-a-lstm-scores`")
print(f"Saved rows: {saved.count():,}")
print(f"Columns:    {saved.columns}")

assert 'lstm_recon_error'     in saved.columns
assert 'lstm_anomaly_flag'    in saved.columns
assert 'lstm_anomaly_flag_v2' in saved.columns
print("\nAll required columns confirmed in Delta table.")

saved.groupBy('lstm_anomaly_flag', 'lstm_anomaly_flag_v2').count().show()
