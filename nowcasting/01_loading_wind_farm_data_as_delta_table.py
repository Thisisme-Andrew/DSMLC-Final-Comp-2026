# Databricks notebook source
# MAGIC %md
# MAGIC ## Wind Farm A

# COMMAND ----------

# DBTITLE 1,Load and Display Wind Farm A Turbine Data from CSV
df_a = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-a/datasets")
display(df_a)

# COMMAND ----------

# DBTITLE 1,Overwrite Wind Farm A as Delta Table
df_a.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-a`")

# COMMAND ----------

# DBTITLE 1,Load and Display Wind Farm A Event Info Data from CSV
df_a_event_info = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-a/event_info.csv")
display(df_a_event_info)

# COMMAND ----------

# DBTITLE 1,Overwrite Wind Farm A Event Info as Delta Format
df_a_event_info.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-a-event-info`")

# COMMAND ----------

# DBTITLE 1,Load and Display Wind Farm A Feature Descriptions from  ...
df_a_feature_description = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-a/feature_description.csv")
display(df_a_feature_description)

# COMMAND ----------

# DBTITLE 1,Overwrite Wind Farm A Feature Descriptions as Delta Format
df_a_feature_description.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-a-feature-description`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wind Farm B

# COMMAND ----------

# DBTITLE 1,Load and Display Wind Farm B Turbine Data from CSV
df_b = spark.read.format("csv") \
    .option("header", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-b/datasets")

display(df_b)

# COMMAND ----------

# DBTITLE 1,Save Wind Farm B DataFrame as Delta Table in Silver
df_b.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-b`")

# COMMAND ----------

df_b_event_info = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-b/event_info.csv")
display(df_b_event_info)

# COMMAND ----------

df_b_event_info.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-b-event-info`")

# COMMAND ----------

df_b_feature_description = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-b/feature_description.csv")
display(df_b_feature_description)

# COMMAND ----------

df_b_feature_description.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-b-feature-description`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wind Farm C

# COMMAND ----------

# DBTITLE 1,Load and Display Wind Farm C Turbine Data from CSV
df_c = spark.read.format("csv") \
    .option("header", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-c/datasets")

display(df_c)

# COMMAND ----------

# DBTITLE 1,Save Wind Farm C DataFrame as Delta Table in Silver Zon ...
df_c.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-c`")

# COMMAND ----------

df_c_event_info = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-c/event_info.csv")
display(df_c_event_info)

# COMMAND ----------

df_c_event_info.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-c-event-info`")

# COMMAND ----------

df_c_feature_description = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load("/Volumes/workspace/wind-turbine-bronze/wind-farm-c/feature_description.csv")
display(df_c_feature_description)

# COMMAND ----------

df_c_feature_description.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("workspace.`wind-turbine-silver`.`wind-farm-c-feature-description`")