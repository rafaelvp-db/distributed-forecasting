# Databricks notebook source
# DBTITLE 0,Notes for Presenter
# MAGIC %md 
# MAGIC # Fine Grained Demand Forecast
# MAGIC This demo is intended for business-aligned personas who are more interested in our ability to enable business outcomes than the exact technical details of how we do it.  For that reason, much of the code in this notebook has been hidden and the logic itself has been streamlined to get right to the points they are concerned with.  If you require a more technically-oriented presentation, please consider using the notebook associated with the [solution accelerator](https://databricks.com/blog/2021/04/06/fine-grained-time-series-forecasting-at-scale-with-facebook-prophet-and-apache-spark-updated-for-spark-3.html).
# MAGIC 
# MAGIC Be sure to select **View: Results Only** before presenting it to customers.

# COMMAND ----------

spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
spark.conf.set("spark.default.parallelism", "10")

# COMMAND ----------

# MAGIC %fs ls /mnt/field-demos/retail/fgforecast/

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
!pip install --upgrade pip && pip install fbprophet

# COMMAND ----------

# MAGIC %md ## Step 1: Examine the Data
# MAGIC 
# MAGIC The dataset from which we wish to generate our forecasts consists of daily sales data for 50 products across 10 store locations for a 5 year period:

# COMMAND ----------

# DBTITLE 1,Review Raw Data
# read the training file into a dataframe
spark.sql("DROP TABLE IF EXISTS hackathon.sales.raw")

sales = spark.read.csv(
  '/mnt/field-demos/retail/fgforecast/train.csv',
  header=True,
  schema="date date, store int, item int, sales int"
)
sales.dropna().write.saveAsTable("hackathon.sales.raw")

# show data
display(spark.read.table("hackathon.sales.raw"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(1) FROM hackathon.sales.raw

# COMMAND ----------

# MAGIC %md As is typical when performing forecasting, we will want to examine the data for trends and seasonality, at both the yearly and weekly levels:

# COMMAND ----------

# DBTITLE 1,View Yearly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM hackathon.sales.raw
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# DBTITLE 1,View Monthly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM hackathon.sales.raw
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# DBTITLE 1,View Weekday Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   extract(dayofweek from date) as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM hackathon.sales.raw
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md ## Step 2: Generate Forecasts
# MAGIC 
# MAGIC There appear to be strong patterns against which we should be able to build a timeseries forecast.  However, with 10 stores and 50 items, we would need to train 500 models to then deliver 500 store-item forecasts:

# COMMAND ----------

# DBTITLE 1,Get DataSet Metrics
# MAGIC %sql -- get dataset metrics
# MAGIC 
# MAGIC SELECT 
# MAGIC   COUNT(DISTINCT store) as stores,
# MAGIC   COUNT(DISTINCT item) as items,
# MAGIC   COUNT(DISTINCT year(date)) as years,
# MAGIC   COUNT(*) as records
# MAGIC FROM hackathon.sales.raw;

# COMMAND ----------

# MAGIC %md ### Forecasts for individual store
# MAGIC The traditional approach is to aggregate the data and to then generate the forecast on the aggregate dataset. 
# MAGIC 
# MAGIC If we aggregate our data across all 10 stores, this will reduce the required number of models down to 50, something that's historically been more viable for our businesses.  To bring this back down to the store level, we can use an allocation based on unit sales: 

# COMMAND ----------

# DBTITLE 1,Generate Allocated Forecasts (Aggregated Stores)
import json
import mlflow
import pandas as pd
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
from pyspark.sql import functions as f

# NOTE 
# ==================================================================================================================
# The general pattern used here is that we sum sales by product across all stores and then generate a forecast
# for that product. The all-store product forecast is then allocated to individual stores based on the percentage of
# historical sales for that product in that location.  This  is representative of how many organizations perform an
# allocated forecast today.
# ==================================================================================================================

experiment_name = "/Users/rafael.pierre@databricks.com/sales_forecast"
experiment_id = ""
if not mlflow.get_experiment_by_name(experiment_name):
  experiment_id = mlflow.create_experiment(name = experiment_name)
else:
  experiment = mlflow.set_experiment(experiment_name = experiment_name)
  experiment_id = experiment.experiment_id

def extract_params(pr_model):
  return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

#train the model using fbProphet
def train_model(history_pd: pd.DataFrame, store: int = None) -> Prophet:
  # configure the model
  
  model = None
  item = history_pd["item"][0]
  if not store:
    store = "all"
  
  min_date = str(history_pd.ds.min())
  max_date = str(history_pd.ds.max())
  run_name = f"run_item_{item}_store_{store}"
  with mlflow.start_run(experiment_id = experiment_id, run_name = run_name) as run:
    model = Prophet(
      interval_width=0.95,
      growth='linear',
      daily_seasonality=False,
      weekly_seasonality=True,
      yearly_seasonality=True,
      seasonality_mode='multiplicative'
    )

    # train the model
    model.fit( history_pd )
    
    # get params
    params = extract_params(model)
    
    # calculate metrics
    metric_keys = ["mse", "mae", "mape"]
    metrics_raw = cross_validation(
        model=model,
        horizon="90 days",
        period="360 days",
        initial = "730 days",
        disable_tqdm=True,
        parallel = "processes"
    )
    cv_metrics = performance_metrics(metrics_raw)
    metrics = {k: cv_metrics[k].mean() for k in metric_keys}
    
    # log metrics, params and model
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.prophet.log_model(
      pr_model = model,
      artifact_path = "model"
    )
  
  return model
  
#predict the next 90 days
def make_prediction(model: Prophet) -> pd.DataFrame:
  future_pd = model.make_future_dataframe(periods=90, 
                                          freq='d', 
                                          include_history=True)
  return model.predict(future_pd)  

  
def forecast_item(history_pd: pd.DataFrame) -> pd.DataFrame:
  #training
  model = train_model(history_pd)
  #predict forecast for the next days
  forecast_pd = make_prediction(model)
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # Add existing real predictions
  forecast_pd['y'] = history_pd['y']
  # get store & item from incoming data set
  forecast_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return forecast_pd[['ds', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

forecast = (
  spark
    .table('hackathon.sales.raw')
    .withColumnRenamed('date','ds')
    .groupBy('item', 'ds')
      .agg(f.sum('sales').alias('y'))
    .orderBy('item','ds')
    .groupBy('item')
      .applyInPandas(forecast_item, schema="ds date, item int, y float, yhat float, yhat_upper float, yhat_lower float")
    .withColumn('training_date', f.current_date()))

# allocation ratios
ratios = spark.sql('''SELECT store, item, sales / SUM(sales) OVER(PARTITION BY item) as ratio FROM (
                        SELECT store, item, SUM(sales) as sales
                        FROM hackathon.sales.raw
                        GROUP BY store, item)''')

results = (
  forecast
    .join(ratios, on='item')
    .withColumn('y',f.expr('y * ratio'))
    .withColumn('yhat',f.expr('yhat * ratio'))
    .selectExpr('ds as date','store','item','y as sales', 'yhat as forecast', 'training_date'))


(results
    .write
    .mode('overwrite')
    .saveAsTable('hackathon.sales.allocated_forecasts', mode = "overwrite")
)

display(spark.table('hackathon.sales.allocated_forecasts'))

# COMMAND ----------

# MAGIC %md ### Forecasts for individual products and store
# MAGIC The alternative available to us through Databricks is to take advantage of the cloud to deliver the 500 individual models we actually require:

# COMMAND ----------

# DBTITLE 1,Generate Fine-Grained Forecasts
# NOTE 
# ========================================================
# The pattern used here is we pull together historical sales
# by store and item combination.  That is then used to produce
# a store-item specific forecast.  A Pandas UDF is used to
# scale-out this work across a Databricks cluster.  The 
# results are captured in a Spark Dataframe which is then
# persisted for later evaluation.
# ========================================================

# retrieve historical data
store_item_history = (
  spark.sql('''SELECT store, item, CAST(date as date) as ds, SUM(sales) as y FROM hackathon.sales.allocated_forecasts
              GROUP BY store, item, ds
              ORDER BY store, item, ds'''))

def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  #training
  store = history_pd["store"][0]
  model = train_model(history_pd, store = store)
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

# MAGIC %md ### Examine Scalability 
# MAGIC This Databricks-enabled approach works for us in that we can decide how many resources we want to allocate for the problem and this directly translates into the time required to complete the operation.  Here you are seeing how different sized environments translate into overall process times for our 500 model runs, but keep in mind that many of our largest customers use this same pattern to complete millions of forecasts within an hour or two daily:
# MAGIC <!-- Test runs on Azure F4s_v2 - 4 cores, 8 GB RAM -->
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/fg-forecast-scalability.png">

# COMMAND ----------

# MAGIC %md Comparing the results of each, we see the fine-grained forecasts deliver the localized variations we were hoping to capture while the allocation method simply returns scaled-variations of the cross-store forecast.  Those variations represent real differences in local demand that we need to fine-tune our operations to to maximize profits:

# COMMAND ----------

# DBTITLE 1,Visualize Allocated Forecasts for Item 1 at Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM hackathon.sales.allocated_forecasts
# MAGIC WHERE item = 1 AND 
# MAGIC       date >= '2018-01-01' AND 
# MAGIC       training_date=current_date()
# MAGIC ORDER BY date, store

# COMMAND ----------

# DBTITLE 1,Visualize Fine Grained Forecasts for Item 1 at Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM hackathon.sales.finegrain_forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY date, store

# COMMAND ----------

# MAGIC %md ## Step 3: Present Data to Analysts
# MAGIC 
# MAGIC While Databricks is great at reducing the time required for us to produce these forecasts, how might analysts consume them. You've already seen the native visualization functionality in these notebooks.  That's a capability intended to assist Data Scientists as they do their work.
# MAGIC 
# MAGIC For analysts, we might leverage [Databricks' SQL Dashboard](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/afcbe07e-3064-4f11-ac13-d71ccc30a9d9-retail-finegrained-forecasting?o=1444828305810485) to present results:

# COMMAND ----------

# MAGIC %md <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_dashboard.PNG' width=800>

# COMMAND ----------

# MAGIC %md We might also wish to present the data through tools like Tableau and Power BI (via native connectors):

# COMMAND ----------

# MAGIC %md <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_powerbi.PNG' width=800>

# COMMAND ----------

# MAGIC %md We can even present these data in Excel (via an ODBC connection):

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_excel_2.PNG' width=800>

# COMMAND ----------


