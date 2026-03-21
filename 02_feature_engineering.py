# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC Wind-turbine predictive maintenance.  Builds rolling, interaction, deviation & scaled features per farm.
# MAGIC
# MAGIC **Output table contract** — every saved table must include at minimum:
# MAGIC `asset_id`, `time_stamp`, `id`, `train_test`, `status_type_id`,
# MAGIC all original `sensor_X_avg` columns,
# MAGIC `sensor_X_roll_mean_1h`, `sensor_X_roll_mean_6h`, `sensor_X_roll_mean_24h`,
# MAGIC `sensor_X_roll_std_6`, `sensor_X_delta`, `sensor_X_deviation`,
# MAGIC `oil_x_vib`, `temp_x_rpm` (Farm A only, if matching sensors exist).

# COMMAND ----------

# CELL 1 — Imports and config

from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType
import pandas as pd, numpy as np
# sklearn no longer needed — scaling done in pure Spark
import warnings; warnings.filterwarnings('ignore')
import pickle, os

CATALOG = "wind-turbine-silver"

FARM_CONFIG = {
    'a': {'table': 'wind-farm-a',
           'output': 'wind-farm-a-features',
           'normal_status': [0]},
    'b': {'table': 'wind-farm-b',
           'output': 'wind-farm-b-features',
           'normal_status': None},   # set after Cell 2
    'c': {'table': 'wind-farm-c',
           'output': 'wind-farm-c-features',
           'normal_status': None},   # set after Cell 2
}

EXCLUDE_COLS = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']

# COMMAND ----------

# CELL 2 — Schema validation

dfs_raw = {}

for farm, cfg in FARM_CONFIG.items():
    table_name = cfg['table']
    df = spark.table(f"`{CATALOG}`.`{table_name}`")
    dfs_raw[farm] = df

    row_count = df.count()
    print(f"\n{'=' * 60}")
    print(f"Farm {farm.upper()} — `{CATALOG}`.`{table_name}`")
    print(f"{'=' * 60}")
    print(f"Row count: {row_count:,}")
    print(f"All columns ({len(df.columns)}): {df.columns}\n")

    required = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Farm {farm} missing required columns: {missing}\n"
            f"Available columns: {df.columns}"
        )
    print("All required columns present")

    print("\nNull counts for key columns:")
    df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in required
    ]).show()

    print(f"Farm {farm.upper()} status_type_id counts:")
    df.groupBy('status_type_id').count().orderBy('status_type_id').show()

    print(f"Farm {farm.upper()} train/test split:")
    df.groupBy('train_test').count().show()

    train_count = df.filter(F.col('train_test') == 'train').count()
    if train_count == 0:
        raise ValueError(f"Farm {farm} has zero train rows — cannot proceed")
    print(f"Farm {farm.upper()} has {train_count:,} train rows")

print("\n>>> Review status_type distributions above.")
print(">>> Update FARM_CONFIG['b']['normal_status'] and FARM_CONFIG['c']['normal_status'] in Cell 3.")

# COMMAND ----------

# CELL 3 — Set normal_status for B/C and define sensor columns per farm

# =====================================================================
# ACTION REQUIRED: uncomment and update with correct values from Cell 2
# =====================================================================
FARM_CONFIG['b']['normal_status'] = [0]
FARM_CONFIG['c']['normal_status'] = [0]

sensor_cols = {}

for farm in ['a', 'b', 'c']:
    df = dfs_raw[farm]
    cols = sorted([
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and not c.endswith(('_max', '_min', '_std'))
    ])
    if len(cols) == 0:
        raise ValueError(f"No sensor columns found in Farm {farm.upper()}")
    sensor_cols[farm] = cols
    avg_ct  = sum(1 for c in cols if c.endswith('_avg'))
    bare_ct = len(cols) - avg_ct
    print(f"Farm {farm.upper()}: {len(cols)} sensors ({avg_ct} _avg, {bare_ct} bare)")

if len(sensor_cols['c']) > 100:
    print(f"\nFarm C has {len(sensor_cols['c'])} sensors — reducing to top 100 by variance")
    train_c = dfs_raw['c'].filter(F.col('train_test') == 'train')
    var_exprs = [F.variance(F.col(c).cast('double')).alias(c)
                 for c in sensor_cols['c']]
    variances = train_c.select(var_exprs).first().asDict()
    sorted_by_var = sorted(
        variances.items(),
        key=lambda kv: kv[1] if kv[1] is not None else 0.0,
        reverse=True,
    )
    sensor_cols['c'] = [kv[0] for kv in sorted_by_var[:100]]
    print(f"Farm C reduced to {len(sensor_cols['c'])} sensors (top by train-set variance)")

# COMMAND ----------

# CELL 4 — Filter to normal-status train rows + ALL test rows

dfs = {}

for farm, cfg in FARM_CONFIG.items():
    df = dfs_raw[farm]
    normal_status = cfg['normal_status']

    if normal_status is None:
        raise ValueError(
            f"Farm {farm}: normal_status is None — update FARM_CONFIG in Cell 3 first"
        )

    df_train = df.filter(
        (F.col('status_type_id').isin(normal_status))
        & (F.col('train_test') == 'train')
    )
    df_test = df.filter(F.col('train_test') == 'test')
    combined = df_train.unionByName(df_test)

    train_ct = df_train.count()
    test_ct  = df_test.count()
    print(
        f"Farm {farm.upper()}: {train_ct:,} normal-status train rows "
        f"+ {test_ct:,} test rows = {train_ct + test_ct:,} total"
    )

    dfs[farm] = combined

# COMMAND ----------

# CELL 5 — Rolling features using Spark Window functions

for farm in ['a', 'b', 'c']:
    df   = dfs[farm]
    cols = sensor_cols[farm]
    print(f"\nFarm {farm.upper()}: computing rolling features for {len(cols)} sensors ...")

    # Cast all sensor columns to double (some farms store values as STRING)
    cast_needed = [c for c in cols if df.schema[c].dataType != DoubleType()]
    if cast_needed:
        print(f"  Casting {len(cast_needed)} non-double columns to DoubleType")
        for c in cast_needed:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    order_window = Window.partitionBy('asset_id').orderBy('time_stamp')
    w6   = order_window.rowsBetween(-5,   0)    # ~1 h   (6 rows × 10-min intervals)
    w36  = order_window.rowsBetween(-35,  0)    # ~6 h
    w144 = order_window.rowsBetween(-143, 0)    # ~24 h

    roll_exprs = []
    for c in cols:
        roll_exprs.extend([
            F.avg(c).over(w6).alias(f"{c}_roll_mean_1h"),
            F.avg(c).over(w36).alias(f"{c}_roll_mean_6h"),
            F.avg(c).over(w144).alias(f"{c}_roll_mean_24h"),
            F.stddev(c).over(w6).alias(f"{c}_roll_std_6"),
            (F.col(c) - F.lag(c, 1).over(order_window)).alias(f"{c}_delta"),
        ])

    df = df.select("*", *roll_exprs)

    new_roll_cols = [
        col_name for col_name in df.columns
        if col_name.endswith((
            '_roll_mean_1h', '_roll_mean_6h',
            '_roll_mean_24h', '_roll_std_6', '_delta',
        ))
    ]

    # Only drop rows where ALL rolling/delta columns are null (first row per turbine)
    all_null_cond = F.lit(True)
    for rc in new_roll_cols:
        all_null_cond = all_null_cond & F.col(rc).isNull()

    before_ct = df.count()
    df = df.filter(~all_null_cond)
    after_ct = df.count()
    print(f"  Dropped {before_ct - after_ct:,} rows where ALL rolling cols are null")

    # Fill remaining partial nulls with 0 to preserve early time-series data
    df = df.fillna(0, subset=new_roll_cols)

    dfs[farm] = df
    print(f"  Final: {after_ct:,} rows, {len(df.columns)} columns")

# COMMAND ----------

# CELL 6 — Interaction terms (Farm A only)

desc_df = spark.table(f"`{CATALOG}`.`wind-farm-a-feature-description`")
desc_pd = desc_df.toPandas()   # small reference table — safe to collect

print(f"Feature description table: {len(desc_pd)} rows\n")

a_columns = set(dfs['a'].columns)

def _resolve_col(name):
    """Match a sensor name to the actual DataFrame column (bare or _avg)."""
    if name in a_columns:
        return name
    avg = f"{name}_avg"
    if avg in a_columns:
        return avg
    return None

# Temperature sensors (exclude ambient)
temp_mask = (
    (desc_pd['description'].str.contains('temp', case=False, na=False)
     | desc_pd['unit'].str.contains('C', case=False, na=False))
    & ~desc_pd['description'].str.contains('ambient', case=False, na=False)
)
temp_sensors = [
    c for c in (
        _resolve_col(s) for s in desc_pd.loc[temp_mask, 'sensor_name']
    ) if c is not None
]

# Vibration / RPM sensors
vib_mask = (
    desc_pd['description'].str.contains('vibration', case=False, na=False)
    | desc_pd['description'].str.contains('rpm', case=False, na=False)
)
vib_sensors = [
    c for c in (
        _resolve_col(s) for s in desc_pd.loc[vib_mask, 'sensor_name']
    ) if c is not None
]

print(f"Temperature sensors: {temp_sensors}")
print(f"Vibration/RPM sensors: {vib_sensors}")

if not temp_sensors:
    print("WARNING: No temperature sensors found — skipping interaction terms")
elif not vib_sensors:
    print("WARNING: No vibration/RPM sensors found — skipping interaction terms")
else:
    dfs['a'] = dfs['a'].withColumn(
        'oil_x_vib',
        F.col(temp_sensors[0]) * F.col(vib_sensors[0]),
    )
    second_vib = vib_sensors[1] if len(vib_sensors) > 1 else vib_sensors[0]
    dfs['a'] = dfs['a'].withColumn(
        'temp_x_rpm',
        F.col(temp_sensors[0]) * F.col(second_vib),
    )
    print("Added interaction columns: oil_x_vib, temp_x_rpm")

# COMMAND ----------

# CELL 7 — Per-turbine baseline deviation (train-only means — NO LEAKAGE)

for farm in ['a', 'b', 'c']:
    df   = dfs[farm]
    cols = sensor_cols[farm]
    print(f"\nFarm {farm.upper()}: computing per-turbine deviation for {len(cols)} sensors ...")

    # Means derived exclusively from train rows
    turbine_means = (
        df.filter(F.col('train_test') == 'train')
          .groupBy('asset_id')
          .agg(*[F.mean(F.col(c)).alias(f"{c}_mean") for c in cols])
    )

    df = df.join(turbine_means, on='asset_id', how='left')

    # Build deviation columns and drop the _mean helpers in a single select
    keep_cols = [F.col(c) for c in df.columns if not c.endswith('_mean')]
    dev_exprs = [
        (F.col(c) - F.col(f"{c}_mean")).alias(f"{c}_deviation")
        for c in cols
    ]
    df = df.select(*keep_cols, *dev_exprs)

    dfs[farm] = df
    print(f"  Added {len(cols)} deviation columns ({df.count():,} rows)")

# COMMAND ----------

# CELL 8 — Standard-scaling (fit on train only, apply to all — pure Spark, no toPandas)

SCALER_DIR = "/tmp/DSMLC-Final-Comp-2026/scalers"
os.makedirs(SCALER_DIR, exist_ok=True)
print(f"Scaler stats will be saved to {SCALER_DIR}")

for farm in ['a', 'b', 'c']:
    df   = dfs[farm]
    cols = sensor_cols[farm]

    roll_cols = [c for c in df.columns if c.endswith((
        '_roll_mean_1h', '_roll_mean_6h',
        '_roll_mean_24h', '_roll_std_6', '_delta',
    ))]
    dev_cols = [c for c in df.columns if c.endswith('_deviation')]

    feature_cols = list(cols) + roll_cols + dev_cols
    if farm == 'a':
        for ic in ['oil_x_vib', 'temp_x_rpm']:
            if ic in df.columns:
                feature_cols.append(ic)

    print(f"\nFarm {farm.upper()}: scaling {len(feature_cols)} feature columns ...")

    df = df.fillna(0, subset=feature_cols)

    # Compute mean and population stddev from train rows entirely in Spark
    train_df = df.filter(F.col('train_test') == 'train')
    train_count = train_df.count()
    if train_count == 0:
        raise ValueError(f"Farm {farm}: no train rows — cannot fit scaler")
    print(f"  Fitting on {train_count:,} train rows")

    agg_exprs = []
    for c in feature_cols:
        agg_exprs.append(F.mean(c).alias(f"{c}__mean"))
        agg_exprs.append(F.stddev_pop(c).alias(f"{c}__std"))

    stats = train_df.select(agg_exprs).first().asDict()

    # Save stats dict for reproducibility
    scaler_path = os.path.join(SCALER_DIR, f"scaler_{farm}.pkl")
    with open(scaler_path, 'wb') as fh:
        pickle.dump(stats, fh)
    print(f"  Scaler stats saved to {scaler_path}")

    # Apply (x - mean) / std via Spark column arithmetic
    feat_set = set(feature_cols)
    col_exprs = []
    for c in df.columns:
        if c in feat_set:
            m = float(stats.get(f"{c}__mean") or 0.0)
            s = float(stats.get(f"{c}__std") or 1.0)
            if s == 0:
                s = 1.0
            col_exprs.append(((F.col(c) - F.lit(m)) / F.lit(s)).alias(c))
        else:
            col_exprs.append(F.col(c))

    df = df.select(*col_exprs)
    dfs[farm] = df
    print(f"  Scaling applied ({df.count():,} rows)")

# COMMAND ----------

# CELL 9 — Output schema validation before saving

REQUIRED_SUFFIXES = [
    '_roll_mean_1h', '_roll_mean_6h', '_roll_mean_24h',
    '_roll_std_6', '_delta', '_deviation',
]

for farm in ['a', 'b', 'c']:
    df = dfs[farm]
    cols_set = set(df.columns)

    print(f"\n{'=' * 60}")
    print(f"Farm {farm.upper()} — output validation")
    print(f"{'=' * 60}")

    for rc in ['asset_id', 'time_stamp', 'id', 'train_test', 'status_type_id']:
        if rc not in cols_set:
            raise ValueError(f"Farm {farm} output missing column: {rc}")

    for sc in sensor_cols[farm]:
        if sc not in cols_set:
            raise ValueError(f"Farm {farm} output missing sensor column: {sc}")

    for sc in sensor_cols[farm]:
        for suffix in REQUIRED_SUFFIXES:
            expected = f"{sc}{suffix}"
            if expected not in cols_set:
                raise ValueError(f"Farm {farm} output missing column: {expected}")

    if farm == 'a':
        for ic in ['oil_x_vib', 'temp_x_rpm']:
            if ic not in cols_set:
                print(f"  WARNING: Farm A missing interaction column '{ic}' "
                      f"(skipped if matching sensors were not found)")

    print(f"  Columns: {len(df.columns)}")
    row_ct = df.count()
    print(f"  Rows:    {row_ct:,}")

    feature_check_cols = [
        c for c in df.columns
        if any(c.endswith(s) for s in REQUIRED_SUFFIXES)
    ]
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in feature_check_cols
    ])
    null_pd = null_counts.toPandas().T
    null_pd.columns = ['null_count']
    nonzero = null_pd[null_pd['null_count'] > 0]

    if nonzero.empty:
        print("  No nulls in any feature column")
    else:
        print("  Null counts in feature columns:")
        print(nonzero.to_string())

    print(f"  Farm {farm.upper()} PASSED validation")

# COMMAND ----------

# CELL 10 — Save to Delta (Spark, not pandas)

for farm, cfg in FARM_CONFIG.items():
    df = dfs[farm]
    output_table = f"`{CATALOG}`.`{cfg['output']}`"
    print(f"\nSaving Farm {farm.upper()} -> {output_table} ...")

    (df.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .partitionBy("asset_id")
       .saveAsTable(output_table))

    saved_ct = spark.table(output_table).count()
    print(f"  Farm {farm.upper()} saved: {saved_ct:,} rows")

print("\nAll farms saved successfully.")
