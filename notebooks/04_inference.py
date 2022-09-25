# Databricks notebook source
import pandas as pd

def predict_udf(history_pd: pd.DataFrame) -> pd.DataFrame:
  
  import mlflow
  from mlflow.tracking.client import MlflowClient
  client = MlflowClient()
  
  model_name = "ForecastingModelUDF"
  registered_model = client.get_registered_model(model_name)
  print(registered_model)
  run_id = registered_model.latest_versions[0].run_id
  model = mlflow.pyfunc.load_model(model_uri = f"runs:/{run_id}/model")
  
  forecast_pd = model.predict(history_pd)
  return forecast_pd

# COMMAND ----------

sales = spark.read.csv(
  '/mnt/field-demos/retail/fgforecast/test.csv',
  header=True,
  schema="id int, date date, store int, item int"
)

spark.sql("drop table if exists hackathon.sales.test_raw")

sales.dropna().select("date", "store", "item").write.saveAsTable(
  "hackathon.sales.test_raw",
)

test_df = spark.sql("select * from hackathon.sales.test_raw")
display(test_df)

# COMMAND ----------

from pyspark.sql import functions as f

spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
spark.conf.set("spark.default.parallelism", "10")

test_df = test_df.withColumnRenamed("date", "ds")
test_df.cache().count()

# generate forecast
results = (
  test_df
    .groupBy('store', 'item')
      .applyInPandas(predict_udf, schema="ds date, store int, item int, yhat float, yhat_upper float, yhat_lower float")
    .withColumnRenamed('yhat','forecast')
    .withColumnRenamed('yhat_upper','forecast_upper')
    .withColumnRenamed('yhat_lower','forecast_lower')
)

spark.sql("drop table if exists hackathon.sales.test_finegrain_forecasts")

(results
    .write
    .mode('overwrite')
    .saveAsTable('hackathon.sales.test_finegrain_forecasts'))

display(spark.table('hackathon.sales.test_finegrain_forecasts').drop('forecast_upper','forecast_lower'))
