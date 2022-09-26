# Databricks notebook source
# DBTITLE 1,Logging our model as a Custom PyFunc Model
import mlflow
from model_wrapper import ForecastStoreItemModel

# Experiment containing our Prophet runs
source_experiment_name = "/Users/rafael.pierre@databricks.com/sales_forecast"
source_experiment = mlflow.get_experiment_by_name(source_experiment_name)
source_experiment_id = source_experiment.experiment_id

# Experiment for PyFunc Model that will be used as UDF
udf_experiment_name = "/Users/rafael.pierre@databricks.com/sales_forecast_udf"
if not mlflow.get_experiment_by_name(udf_experiment_name):
  mlflow.create_experiment(udf_experiment_name)
experiment = mlflow.get_experiment_by_name(udf_experiment_name)
experiment

# COMMAND ----------

with mlflow.start_run(
  experiment_id = experiment.experiment_id,
  run_name = "store_item_pyfunc"
) as run:
  
  model_info = mlflow.pyfunc.log_model(
    artifact_path = "model",
    code_path = ["model_wrapper.py"],
    python_model = ForecastStoreItemModel(source_experiment_id),
    pip_requirements = ["fbprophet"]
  )

# COMMAND ----------

model_version = mlflow.register_model(
  model_uri = model_info.model_uri, name = "ForecastingModelUDF"
)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

tags = {
  "udf": "true",
  "reviewed": "false",
  "store": "true",
  "item": "true",
  "schema": "date date, item int, yhat float, yhat_upper float, yhat_lower float"
}

for key in tags.keys():
  client.set_model_version_tag(
    name = model_version.name,
    version = model_version.version,
    key = key,
    value = tags[key]
  )

# COMMAND ----------

del ForecastStoreItemModel

# COMMAND ----------


