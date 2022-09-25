# Databricks notebook source
import mlflow

experiment_id = "084f576022a3421eae39b919a396e050"
runs = mlflow.search_runs(
  experiment_ids = [experiment_id],
  filter_string = "tags.mlflow.runName = 'store_item_udf' AND status = 'FINISHED'"
)
runs.sort_values(by = "end_time", ascending = False)

# COMMAND ----------

run_id = runs.sort_values(by = "end_time", ascending = False).loc[0, "run_id"]
print(run_id)
pyfunc_udf = mlflow.pyfunc.load_model(model_uri = f"runs:/{run_id}/model")

# COMMAND ----------

pyfunc_udf.metadata.get_model_info()

# COMMAND ----------

# retrieve historical data
store_item_history = (
  spark.sql('''SELECT store, item, CAST(date as date) as ds, SUM(sales) as y FROM hackathon.sales.allocated_forecasts
              GROUP BY store, item, ds
              ORDER BY store, item, ds'''))

# generate forecast
results = pyfunc_udf.predict(model_input = store_item_history)

# COMMAND ----------

grouped_df = store_item_history.groupBy('store', 'item')
  
grouped_df.apply

# COMMAND ----------


