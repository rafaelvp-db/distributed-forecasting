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

def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # fetch the model for the particular store/item combination
  model = mlflow.
  #predict forecast for the next days
  forecast_pd = make_prediction(model)
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  forecast_pd['y'] = history_pd['y']
  # get store & item from incoming data set
  forecast_pd['store'] = history_pd['store'].iloc[0]
  forecast_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return forecast_pd[['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

# generate forecast
results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema="ds date, store int, item int, y float, yhat float, yhat_upper float, yhat_lower float")
    .withColumn('training_date', f.current_date() )
    .withColumnRenamed('ds','date')
    .withColumnRenamed('y','sales')
    .withColumnRenamed('yhat','forecast')
    .withColumnRenamed('yhat_upper','forecast_upper')
    .withColumnRenamed('yhat_lower','forecast_lower'))


(results
    .write
    .mode('overwrite')
    .saveAsTable('hackathon.sales.finegrain_forecasts'))

display(spark.table('hackathon.sales.finegrain_forecasts').drop('forecast_upper','forecast_lower'))

# COMMAND ----------


