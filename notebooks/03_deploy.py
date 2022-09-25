# Databricks notebook source
# DBTITLE 1,Logging our model as a Custom PyFunc Model
import mlflow
from model_wrapper import ForecastStoreItemModel

experiment = mlflow.get_experiment_by_name("/Users/rafael.pierre@databricks.com/sales_forecast")
experiment

# COMMAND ----------

with mlflow.start_run(run_name = "store_item_udf") as run:
  
  mlflow.pyfunc.log_model(
    artifact_path = "model",
    code_path = ["model_wrapper.py"],
    python_model = ForecastStoreItemModel(experiment.experiment_id),
    pip_requirements = ["fbprophet"]
  )

# COMMAND ----------


