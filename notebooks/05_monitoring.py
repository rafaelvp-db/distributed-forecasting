# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select * from hackathon.sales.test_finegrain_forecasts

# COMMAND ----------

spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")

# COMMAND ----------

spark.conf.get("spark.databricks.optimizer.adaptive.enabled")

# COMMAND ----------


