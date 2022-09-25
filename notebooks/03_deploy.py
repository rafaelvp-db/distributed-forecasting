# Databricks notebook source
import mlflow

experiment_id = "8ec4b771e8254f128678a40b5bc63b83"
experiment = mlflow.get_experiment(experiment_id)
runs = mlflow.search_runs(experiment_ids = [experiment_id], filter_string = "status = 'FINISHED'")

runs.head()

# COMMAND ----------

# DBTITLE 1,Filtering only granular forecasting model runs
filtered_runs = runs[runs["tags.mlflow.runName"].str.match(r"(run_item_[0-9]+_store_[0-9]+)")]
filtered_runs.loc[:, ["run_id", "tags.mlflow.runName"]].sample(5)

# COMMAND ----------

# DBTITLE 1,Logging our model as a Custom PyFunc Model
from model_wrapper import ForecastStoreItemModel

with mlflow.start_run(run_name = "store_item_udf") as run:
  
  mlflow.pyfunc.log_model(
    artifact_path = "model",
    code_path = ["model_wrapper.py"],
    python_model = ForecastStoreItemModel(experiment_id),
    pip_requirements = ["fbprophet"]
  )

# COMMAND ----------


