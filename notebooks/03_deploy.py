# Databricks notebook source
import mlflow

# COMMAND ----------

experiment_id = "8ec4b771e8254f128678a40b5bc63b83"
experiment = mlflow.get_experiment(experiment_id)
runs = mlflow.search_runs(experiment_ids = [experiment_id])

runs

# COMMAND ----------

# DBTITLE 1,Filtering only granular forecasting model runs
filtered_runs = runs[runs["tags.mlflow.runName"].str.match(r"(run_item_[0-9]+_store_[0-9]+)")]
filtered_runs.loc[:, ["run_id", "tags.mlflow.runName"]]

# COMMAND ----------

import pandas as pd
import time

def register_models(df_runs: pd.DataFrame):

  for _, row in df_runs.iterrows():
    time.sleep(0.2) #gently target the API
    run = mlflow.get_run(row["run_id"])

    time.sleep(0.2) #gently target the API
    mlflow.register_model(
      model_uri = f"runs:/{run.info.run_id}/model",
      name = f"model_{row['tags.mlflow.runName']}",
      await_registration_for=5
    )
    
register_models(filtered_runs)

# COMMAND ----------


