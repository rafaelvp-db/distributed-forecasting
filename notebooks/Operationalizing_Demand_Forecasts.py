# Databricks notebook source
# MAGIC %md ---
# MAGIC title: Scaling Forecast Generation with Databricks - Part 2
# MAGIC authors:
# MAGIC - Bryan Smith, Bilal Obeidat, Rob Saker
# MAGIC tags:
# MAGIC - prophet
# MAGIC - fbprophet
# MAGIC - pandas
# MAGIC - udf
# MAGIC - retail
# MAGIC - manufacturing
# MAGIC - supply-chain
# MAGIC - forecast
# MAGIC - forecasting
# MAGIC - time-series
# MAGIC - demand
# MAGIC - python
# MAGIC - sql
# MAGIC created_at: 2020-03-15
# MAGIC updated_at: 2020-04-16
# MAGIC tldr: To demonstrate how multiple, independent forecasts can be generated at scale
# MAGIC thumbnail: https://databricks.com/wp-content/uploads/2020/01/time-series-forecasting-prophet-prediction-chart.jpg
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install fbprophet

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/8241189](https://demo.cloud.databricks.com/#notebook/8241189)

# COMMAND ----------

# MAGIC  %md #Scaling Forecast Generation with Databricks - Part 2
# MAGIC  
# MAGIC This notebook continues building on concepts that were explored in Part 1 of this demo.  Information on the loading of the required dataset was also included in that notebook.  Please review Part 1 of this demo before proceeding.
# MAGIC 
# MAGIC In the last notebook, we looked at strategies for generating fine-grained forecasts at scale.  We made use of a popular forecasting library, [FBProphet](https://facebook.github.io/prophet/), on a Databricks 6.0+ cluster and a publically available [dataset](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) with 5-years of store-item sales data.  Let's configure our session to continue building on this work:

# COMMAND ----------

# configure distribution target for groupBy
spark.conf.set("spark.sql.shuffle.partitions", 500 ) 
# the hardcoded value is based on our knowledge of the data
# this is not necessarily a best practice in all scenarios

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/Retail/demand_forecast/forecasting_banner3.png)

# COMMAND ----------

# MAGIC %md Now that we have a preferred strategy in mind, it's time to explore how we might operationalize the generation of forecasts in our environment. Requirements for this will vary based on the specific retail-scenario we are engaged in. We need to consider how fast products move, how frequently we have access to new and updated data, how reliable our forecasts remain over time (in the face of new data), and the lead times we require to respond to any insights derived from our forecasts.  But given the scalable patterns we explored in the last lab, processing time for forecast generation is not a limiting factor that should require us to compromise our work.

# COMMAND ----------

# MAGIC %md ## Setting Up the Simulation
# MAGIC 
# MAGIC For our demonstration, let's say that we've decided to generate new forecasts every day for each of our store-item combinations.  In this scenario, we might expect that we receive new sales data on a daily basis and that we use that new data in combination with our other historical data to generate forecasts at the start of the next day.
# MAGIC 
# MAGIC The dataset with which we are working has data from **January 1, 2013 through December 31, 2017**.  To simulate our movement through time, we will start our demonstration on **December 1, 2017**, generating forecasts using data from the start of 2013 up to **November 30, 2017**, the day prior, and then repeat this process for each day through **January 1, 2018**, the last day for which we have prior day historical values.
# MAGIC 
# MAGIC <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/Retail/demand_forecast/dec2017_calendar.png" width=300>
# MAGIC 
# MAGIC To assist with this movement through time, let's generate a date range object, the members of which we can iterate over:

# COMMAND ----------

# DBTITLE 1,Define Date Range
from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
print( 
  list(daterange( date(2017,12,1), date(2018,1,1)))
  )

# COMMAND ----------

# MAGIC %md Now we need a function which we can apply to our Spark DataFrame in order to generate our forecast.  We'll use the function we employed in our last demonstration:
# MAGIC 
# MAGIC **NOTE** Prophet will complain of *ERROR:fbprophet:Importing plotly failed. Interactive plots will not work* when it is imported.  This will not effect our work and can be ignored.

# COMMAND ----------

# DBTITLE 1,Define Training & Forecasting Function
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

from fbprophet import Prophet

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

# structure of udf result set
result_schema =StructType([
  StructField('store',IntegerType()),
  StructField('item',IntegerType()),
  StructField('ds',DateType()),
  StructField('trend',FloatType()),
  StructField('yhat_lower', FloatType()),
  StructField('yhat_upper', FloatType()),
  StructField('trend_lower', FloatType()),
  StructField('trend_upper', FloatType()),
  StructField('multiplicative_terms', FloatType()),
  StructField('multiplicative_terms_lower', FloatType()),
  StructField('multiplicative_terms_upper', FloatType()),
  StructField('weekly', FloatType()),
  StructField('weekly_lower', FloatType()),
  StructField('weekly_upper', FloatType()),
  StructField('yearly', FloatType()),
  StructField('yearly_lower', FloatType()),
  StructField('yearly_upper', FloatType()),
  StructField('additive_terms', FloatType()),
  StructField('additive_terms_lower', FloatType()),
  StructField('additive_terms_upper', FloatType()),
  StructField('yhat', FloatType())
  ])

# get forecast
@pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def get_forecast_spark(keys, grouped_pd):
  
  # drop nan records
  grouped_pd = grouped_pd.dropna()
  
  # identify store and item
  store = keys[0]
  item = keys[1]
  days_to_forecast = keys[2]
  
  # configure model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # train model
  model.fit( grouped_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]  )
  
  # make forecast
  future_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=True
    )
  
  # retrieve forecast
  forecast_pd = model.predict( future_pd )
  
  # assign store and item to group results
  forecast_pd['store']=store
  forecast_pd['item']=item
  
  # return results
  return forecast_pd[
      ['store', 'item', 'ds', 'trend', 'yhat_lower', 'yhat_upper',
       'trend_lower', 'trend_upper', 'multiplicative_terms',
       'multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',
       'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
       'yearly_upper', 'additive_terms', 'additive_terms_lower',
       'additive_terms_upper', 'yhat']
        ]

# COMMAND ----------

# MAGIC %md Because we are automating the generation of forecasts, it is important we capture some data with which we can evaluate their reliability. Commonly used metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Mean Percent Error (MPE).  We will calculate all of these for each store-item forecast. To do this, we will need another function that we will apply to our forecast dataset:

# COMMAND ----------

# DBTITLE 1,Define Forecast Evaluation Function
from sklearn.metrics import mean_squared_error, mean_absolute_error

from math import sqrt

import numpy as np
import pandas as pd

# define evaluation metrics generation function
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType()),
  StructField('mae', FloatType())
  ])

@pandas_udf( eval_schema, PandasUDFType.GROUPED_MAP )
def evaluate_forecast( keys, forecast_pd ):
  
  # get store & item in incoming data set
  training_date = keys[0]
  store = int(keys[1])
  item = int(keys[2])
  
  # MSE & RMSE
  mse = mean_squared_error( forecast_pd['y'], forecast_pd['yhat'] )
  rmse = sqrt(mse)
  
  # MAE
  mae = mean_absolute_error( forecast_pd['y'], forecast_pd['yhat'] )
  
  # assemble result set
  results = {
    'training_date':[training_date], 'store':[store], 'item':[item], 
    'mse':[mse], 'rmse':[rmse], 'mae':[mae]
    }
  return pd.DataFrame( data=results )

# COMMAND ----------

# MAGIC %md Now we are just about ready to go.  The last thing we need to do is read our historical dataset into a Spark DataFrame:

# COMMAND ----------

# DBTITLE 1,Read Historical Data
from pyspark.sql.types import *

# structure of the training data set
history_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# read the training file into a dataframe
history = spark.read.csv(
  '/mnt/databricks-datasets-private/Retail/demand_forecast/train/train.csv', 
  header=True, 
  schema=history_schema
  ).cache()

# make the dataframe queriable as a temporary view
history.createOrReplaceTempView('history')

# COMMAND ----------

# MAGIC %md ## Running the Simulation
# MAGIC 
# MAGIC With everything in place, we can simulate the generation of forecasts on a daily basis.  We will loop through the dates December 1, 2017 through January 1, 2018, generate our forecast and evaluation metrics for each, and then save this off to storage for later consumption:
# MAGIC 
# MAGIC **NOTE** We are storing the data to the /tmp directory within Databricks.  This is not a recommended best practice and is being done here solely for the purpose of enabling demonstration.

# COMMAND ----------

# DBTITLE 1,Train, Forecast, Evaluate & Store Results
from pyspark.sql.functions import lit

# increment the "current date" for which we are training
for i, training_date in enumerate( daterange(date(2017,12,1), date(2018,1,1)) ):
  
  print(date.strftime(training_date, '%Y-%m-%d'))
  
  # get data up to the day prior to the training date
  sql_statement = '''
    SELECT
      store,
      item,
      date,
      SUM(sales) as sales
    FROM history
    WHERE date < '{0}'
    GROUP BY store, item, date
    ORDER BY store, item, date
    '''.format(training_date)
  store_item_history = spark.sql( sql_statement )
  
  # generate forecast for this data
  forecasts = (
    store_item_history
      .groupBy('store', 'item', lit(30).alias('days_to_forecast'))
      .apply(get_forecast_spark)
      .withColumn('training_date', lit(date.strftime(training_date, '%Y-%m-%d')).cast(DateType()) )
      ).cache() 
  
  # combine forecasts with actuals
  forecast_actuals = (
    forecasts
      .filter('ds < \'{0}\''.format(date.strftime(training_date, '%Y-%m-%d')))       # limit dataset to periods with both history and actuals
      .join(store_item_history, forecasts.ds==store_item_history.date, 'left_outer')
      .select(forecasts.training_date, forecasts.store, forecasts.item, store_item_history.sales.cast(FloatType()).alias('y'), forecasts.yhat)
    )
  
  # calculate metrics comparing forecasts to actuals
  eval_metrics = (
    forecast_actuals      
      .select('training_date', 'store', 'item', 'y', 'yhat')
      .groupBy('training_date', 'store', 'item')
      .apply(evaluate_forecast)
      )  
  
  # save data to delta (reset storage if first run in simulation)
  if i==0:
    spark.sql('drop table if exists forecasts')
    forecasts.write.format('delta').partitionBy('training_date').mode('overwrite').save('/tmp/forecasts')
    spark.sql('drop table if exists forecast_evals')
    eval_metrics.write.format('delta').partitionBy('training_date').mode('overwrite').save('/tmp/forecast_evals')
  else:
    forecasts.write.format('delta').partitionBy('training_date').mode('append').save('/tmp/forecasts')
    eval_metrics.write.format('delta').partitionBy('training_date').mode('append').save('/tmp/forecast_evals')   



# COMMAND ----------

# MAGIC %md ## Examining the Results
# MAGIC 
# MAGIC At this point our forecasts have been generated and evaluation metrics for each forecast run, *i.e.* each store-item combination forecasted on a given training date, have been derived.  These data have been persisted to Delta Lake, and by defining tables against the stored data, we can now examine the results of our efforts:

# COMMAND ----------

# DBTITLE 1,Define Queryable Access to Data
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS forecasts
# MAGIC   USING DELTA
# MAGIC   LOCATION '/tmp/forecasts';
# MAGIC   
# MAGIC CREATE TABLE IF NOT EXISTS forecast_evals
# MAGIC   USING DELTA
# MAGIC   LOCATION '/tmp/forecast_evals';

# COMMAND ----------

# MAGIC %md Here, for example, is our forecast for item 1 as generated on our simulated January 1, 2018 training date:

# COMMAND ----------

# DBTITLE 1,Retrieve Item 1 Forecast
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   ds,
# MAGIC   store,
# MAGIC   yhat,
# MAGIC   yhat_upper,
# MAGIC   yhat_lower
# MAGIC FROM forecasts
# MAGIC WHERE
# MAGIC   training_date='2017-12-01' AND
# MAGIC   ds > training_date AND
# MAGIC   item=1
# MAGIC ORDER BY ds, store;

# COMMAND ----------

# MAGIC %md And here are the evaluation metrics for the model that generated this forecast:

# COMMAND ----------

# DBTITLE 1,Retrieve Item 1 Forecast Evaluation Metrics
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM forecast_evals
# MAGIC WHERE
# MAGIC   training_date='2017-12-01' AND
# MAGIC   item=1
# MAGIC ORDER BY store;

# COMMAND ----------

# MAGIC %md But because our analysts don't want to consume this data through SQL statements in a notebook, let's construct a view that simplifies the dataset and makes it easier to consume in tools like PowerBI and Tableau:

# COMMAND ----------

# DBTITLE 1,Define User-Friendly View
# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS forecasts_for_pbi;
# MAGIC 
# MAGIC CREATE TEMPORARY VIEW forecasts_for_pbi AS
# MAGIC   SELECT
# MAGIC     a.training_date,
# MAGIC     a.date,
# MAGIC     a.store,
# MAGIC     a.item,
# MAGIC     (CASE WHEN b.date < a.training_date THEN b.sales ELSE NULL END) as actuals,
# MAGIC     a.yhat as forecast,
# MAGIC     a.yhat_upper as forecast_upper,
# MAGIC     a.yhat_lower as forecast_lower,
# MAGIC     (CASE WHEN b.date < a.training_date THEN 1 ELSE 0 END) as IsHistory
# MAGIC   FROM ( -- last 30 days worth of forecasts
# MAGIC     SELECT
# MAGIC       training_date,
# MAGIC       ds as date,
# MAGIC       store,
# MAGIC       item,
# MAGIC       yhat,
# MAGIC       yhat_upper,
# MAGIC       yhat_lower
# MAGIC     FROM forecasts
# MAGIC     ) a
# MAGIC   LEFT OUTER JOIN ( -- history for last 30 days
# MAGIC     SELECT
# MAGIC       date,
# MAGIC       store,
# MAGIC       item,
# MAGIC       sales
# MAGIC     FROM history
# MAGIC     WHERE date >= DATE_ADD((SELECT MIN(training_date) FROM forecasts),-30)
# MAGIC     ) b
# MAGIC     ON 
# MAGIC       a.date=b.date AND 
# MAGIC       a.item=b.item AND 
# MAGIC       a.store=b.store

# COMMAND ----------

# MAGIC %md Using this view, we can connect to our data using a tool such as Power BI to generate interactive reports that make consumption and interpretation of the forecast data easier for our analysts:
# MAGIC 
# MAGIC ![](https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/Retail/demand_forecast/pbi_forecasts.png)
