# Databricks notebook source
# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/model-monitoring/e7394149-4c48-42a3-94af-11cee8964415/databricks_model_monitoring-0.0.2-py3-none-any.whl"

# COMMAND ----------



# COMMAND ----------

model_name = "ForecastingModelUDF"
monitor_name = model_name + "_monitor" 
model_stage = "Staging"
monitor_db = "forecastmonitordb"
spark.sql(f"create database if not exists {monitor_db}")

print(f"monitor_name={monitor_name}, monitor_db={monitor_db}, model_name={model_name}")

# COMMAND ----------

def cleanup_existing_monitor(monitor_db, model_name, monitor_name):
  try:
    mm_info = mm.get_monitor_info(model_name=model_name, monitor_name=monitor_name)
  except:
    print("No existing monitor to clean up...")
    return
  print(f"Deleting existing monitor for {model_name}")
  mm.delete_monitor(model_name=model_name, monitor_name=monitor_name)

def cleanup_registered_model(registry_model_name):
  """
  Utilty function to delete a registered model in MLflow model registry.
  To delete a model in the model registry all model versions must first be archived.
  This function thus first archives all versions of a model in the registry prior to
  deleting the model
  
  :param registry_model_name: (str) Name of model in MLflow model registry
  """
  client = MlflowClient()

  filter_string = f'name="{registry_model_name}"'

  model_versions = client.search_model_versions(filter_string=filter_string)
  
  if len(model_versions) > 0:
    print(f"Deleting following registered model: {registry_model_name}")
    
    # Move any versions of the model to Archived
    for model_version in model_versions:
      try:
        model_version = client.transition_model_version_stage(name=model_version.name,
                                                              version=model_version.version,
                                                              stage="Archived")
      except mlflow.exceptions.RestException:
        pass

    client.delete_registered_model(registry_model_name)
    
  else:
    print("No registered models to delete")

# COMMAND ----------

import databricks.model_monitoring as mm


# Create monitor
mm_info = mm.create_monitor(
  model_name=model_name,
  model_type="classifier",
  logging_schema=logging_schema,
  prediction_col="prediction",
  label_col=target_col,
  granularities=["1 day"], 
  id_cols=keys, 
  slicing_exprs=["seniorCitizen", "totalCharges > 1000"],
  database_name=monitor_db,
  monitor_name=monitor_name
 )

print(mm_info)

# COMMAND ----------


