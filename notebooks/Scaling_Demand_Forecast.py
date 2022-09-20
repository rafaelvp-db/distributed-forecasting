# Databricks notebook source
# MAGIC %md ---
# MAGIC title: Scaling Forecast Generation with Databricks - Part 1
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

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/6494747](https://demo.cloud.databricks.com/#notebook/6494747)

# COMMAND ----------

# MAGIC  %md #Scaling Forecast Generation with Databricks - Part 1
# MAGIC 
# MAGIC The objective of this notebook is to illustrate how we might generate a large number of fine-grained forecasts at the store-item level in an efficient manner leveraging the distributed computational power of Databricks.  For more information on this topic, be sure to check out [this blog](https://databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html) and its related notebook. (For even more forecasting examples including those that levarage alternative libraries along with event data and weather regressors, please see [this additional post](https://databricks.com/blog/2020/03/26/new-methods-for-improving-supply-chain-demand-forecasting.html))
# MAGIC 
# MAGIC For this exercise, we will make use of an increasingly popular library for demand forecasting, [FBProphet](https://facebook.github.io/prophet/), which we will load into the notebook session associated with a cluster running Databricks 6.0 or higher:

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/Retail/demand_forecast/forecasting_banner3.png)

# COMMAND ----------

# MAGIC %md ## Examine the Data
# MAGIC 
# MAGIC For our training dataset, we will make use of 5-years of store-item unit sales data for 50 items across 10 different stores.  This data set is publicly available as part of a past Kaggle competition and can be downloaded [here](https://www.kaggle.com/c/demand-forecasting-kernels-only/data). Once downloaded, we can uzip the *train.csv.zip* file and upload the decompressed CSV to */FileStore/tables/demand_forecast/train/* using the file import steps documented [here](https://docs.databricks.com/data/tables.html#create-table-ui). Please note when performing the file import, you don't need to select the *Create Table with UI* or the *Create Table in Notebook* options to complete the import process.
# MAGIC 
# MAGIC With the dataset accessible within Databricks, we can now explore it in preparation for modeling:

# COMMAND ----------

# MAGIC %pip install fbprophet

# COMMAND ----------

# DBTITLE 1,Read the Historical Data
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
  )

# make the dataframe queriable as a temporary view
history.createOrReplaceTempView('history')
#display(history)

# COMMAND ----------

# MAGIC %md When performing demand forecasting, we are often interested in general trends and seasonality.  Let's start our exploration by examing the annual trend in unit sales:

# COMMAND ----------

# DBTITLE 1,Annual Trend
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM history
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md It's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast.  But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, weeks or months from our last observation, we might expect continued linear growth over that time span.
# MAGIC 
# MAGIC Now let's examine seasonality.  If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:

# COMMAND ----------

# DBTITLE 1,Annual Seasonality
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM history
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1) and then a steady pickup over the week heading back to the Sunday high.  This pattern seems to be pretty stable across the five years of observations, increasing with scale given the overall annual trend in sales growth:

# COMMAND ----------

# DBTITLE 1,Weekly Seasonality
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   DAYOFWEEK(date) as weekday,
# MAGIC   --CONCAT(DATE_FORMAT(date, 'u'), '-', DATE_FORMAT(date, 'EEEE')) as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, DAYOFWEEK(date) --, CONCAT(DATE_FORMAT(date, 'u'), '-', DATE_FORMAT(date, 'EEEE'))
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md But as mentioned at the top of this section, there are 10 stores represented in this dataset, and while these stores seem to behave in similar manners, there is still some variation between them as can be seen with a close examination of average monthly sales between stores:

# COMMAND ----------

# DBTITLE 1,Month of Year Sales, All Items, Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   MONTH(month) as month,
# MAGIC   store,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     store,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   GROUP BY TRUNC(date, 'MM'), store
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), store
# MAGIC ORDER BY month, store;

# COMMAND ----------

# MAGIC %md When we examine this pattern at a per-item level, we see more striking differences between the stores emerge:

# COMMAND ----------

# DBTITLE 1,Monthly Sales, Item 1, Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   MONTH(month) as month,
# MAGIC   store,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     store,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   WHERE item=1
# MAGIC   GROUP BY TRUNC(date, 'MM'), store
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), store
# MAGIC ORDER BY month, store;

# COMMAND ----------

# MAGIC %md And this reveals the fundamental issue we wish to tackle in this demonstration.  When we examine our data at an aggregate level, i.e. summing sales across stores or individual products, the aggregation hides what could be some useful variability in our data.  By analyzing our data at the store-item level, we hope to be able to construct forecasts capable of picking up on these subtle differences in behaviours.

# COMMAND ----------

# MAGIC %md ## Generating Store-Item Level Forecasts Sequentially
# MAGIC 
# MAGIC With this in mind, we will attempt to generate store-item level forecasts as we would on a standard machine without access to distributed data processing capabilities. To get us started, we will create a distinct list of store-item combinations:

# COMMAND ----------

# DBTITLE 1,Get Store-Item Combinations
# retrieve distinct store-item combinations for which to produce forecasts
store_items = (
  history
    .select('store', 'item')
    .distinct()
    .collect()
  )

# COMMAND ----------

# MAGIC %md We will iterate over each store-item combination and for each, extract a pandas DataFrame representing the subset of historical data for that store and item.  On that subset-DataFrame, we will train a model and generate a forecast.  To facilitate this, we will write a simple function to receive the subset-DataFrame and return the forecast:
# MAGIC 
# MAGIC **NOTE** Prophet will complain of *ERROR:fbprophet:Importing plotly failed. Interactive plots will not work* when it is imported.  This will not effect our work and can be ignored.

# COMMAND ----------

# DBTITLE 1,Define Training & Forecasting Function
import pandas as pd

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

from fbprophet import Prophet

# train model & generate forecast from incoming dataset
def get_forecast_sequential(store_item_pd, days_to_forecast):
  
  # retrieve store and item from dataframe
  store = store_item_pd['store'].iloc[0]
  item = store_item_pd['item'].iloc[0]
  
  # configure model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # fit to dataset
  model.fit( 
    store_item_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']] 
    )
  
  # make forecast
  future_store_item_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=True
    )
  
  # retrieve forecast
  forecast_store_item_pd = model.predict( future_store_item_pd )
  
  # assemple result set 
  forecast_store_item_pd['store']=store # assign store field
  forecast_store_item_pd['item']=item # assign item field
  
  # return forecast
  return forecast_store_item_pd[ 
      ['store', 'item', 'ds', 'trend', 'yhat_lower', 'yhat_upper',
       'trend_lower', 'trend_upper', 'multiplicative_terms',
       'multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',
       'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
       'yearly_upper', 'additive_terms', 'additive_terms_lower',
       'additive_terms_upper', 'yhat']
       ]

# COMMAND ----------

# MAGIC %md Now we can iterate over these store-item combinations, generating a forecast for each:

# COMMAND ----------

# DBTITLE 1,Train & Forecast Sequentially
# assemble historical dataset
history_pd = history.toPandas()

# for each store-item combination:
for i, store_item in enumerate(store_items):
    
  # extract data subset for this store and item
  store_item_history_pd = history_pd[ 
    (history_pd['store']==store_item['store']) & 
    (history_pd['item']==store_item['item']) 
    ].dropna()
  
  # fit model on store-item subset and produce forecast
  store_item_forecast_pd = get_forecast_sequential(store_item_history_pd, days_to_forecast=30)
   
  # concatonate forecasts to build a single resultset
  if i>0:
    store_item_accum_pd = store_item_accum_pd.append(store_item_forecast_pd, sort=False)
  else:
    store_item_accum_pd = store_item_forecast_pd

# display some results on screen
store_item_accum_pd.head()

# COMMAND ----------

# MAGIC %md While this produces the results we're after, the time it's taking to do so is not supportable. The reliability of timeseries forecasts degrades over time.  We need to generate forecasts far enough out to be actionable but not so far out that they are untrustworthy. For many organizations, this means we need generate weekly, daily or even hourly forecasts based on the latest historical data.  When we consider that a given store may contain 10s or 100s of 1000s of products and then multiply this by the number of stores for which we are generating forecasts, we must have a means to scale this work; processing that volume of data sequentially just isn't a viable solution.

# COMMAND ----------

# MAGIC %md ## Generating Store-Item Level Forecasts in Parallel
# MAGIC 
# MAGIC Leveraging Spark and Databricks, we can easily solve this problem.  Instead of iterating over the set of store-item combinations, we will simply group our data by store and item, forcing store-item combinations to be partitioned across the resources in our cluster. To each store-item grouping, we will apply a function, similar to what we did before, to generate a forecast for each combination. The result will be a unified dataset, addressable as a Spark DataFrame.
# MAGIC 
# MAGIC To get us started, let's re-write our forecast-generating function so that it may be applied to a Spark DataFrame. What you'll notice is that we are defining this function as a [pandas UDF](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) which enables the efficient application of pandas functionality to grouped data in a Spark DataFrame.  But despite the slightly different function signature (which requires us to pre-define the structure of the pandas DataFrame that this function will produce), the internal logic is largely the same as the previous function:

# COMMAND ----------

# DBTITLE 1,Define Training & Forecasting Function for Spark
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

# MAGIC %md With our function defined, we can now group our data and apply the function to each group to generate a store-item forecast:

# COMMAND ----------

# DBTITLE 1,Train & Forecast in Parallel with Spark
from pyspark.sql.functions import lit

# configure distribution target for groupBy
spark.conf.set("spark.sql.shuffle.partitions", 500 ) 
# the hardcoded value is based on our knowledge of the data
# this is not necessarily a best practice in all scenarios

# generate forecasts
# the days_to_forecast field is used to overcome inability to pass params to pandas udf
store_item_accum_spark = (
  history
    .groupBy('store','item', lit(30).alias('days_to_forecast'))
    .apply(get_forecast_spark)
  ).cache()

# action to trigger end-to-end processing (for fair comparison to above)
store_item_accum_spark.count()

# display some results on screen
display(store_item_accum_spark)

# COMMAND ----------

# MAGIC %md The timings achieved by parallelizing the workload are much better than those achieved with sequential process.  What's nice is that as we add resources to the cluster, we can reduce the overall time required to perform the work.  Here are some sample timings achieved by running the above process on different sized clusters: 
# MAGIC 
# MAGIC | Worker VM Type |Cores per VM| Worker Count | Total Worker Cores |  Timing |
# MAGIC |----------------|------------|--------------|--------------------|---------|
# MAGIC | DS3 v2         | 4          | 2            |  8                 | 5.41 m  |
# MAGIC | DS3 v2         | 4          | 3            | 12                 | 3.03 m  |
# MAGIC | DS3 v2         | 4          | 4            | 16                 | 2.83 m  |
# MAGIC | DS3 v2         | 4          | 6            | 24                 | 1.65 m  |
# MAGIC | DS3 v2         | 4          | 8            | 32                 | 1.36 m  |
# MAGIC 
# MAGIC Graphing this data, you can see the general trend as we scale-out the work:

# COMMAND ----------

# DBTITLE 1,Examine Performance vs. Scale
display(
  spark.createDataFrame( 
    [(8, 5.41), (12, 3.03), (16, 2.83), (24, 1.65), (32, 1.36)], 
    schema=['cores', 'minutes']
    )
  )

# COMMAND ----------

# MAGIC %md ## Comparing Fine-Grain Forecasting to an Allocation Strategy
# MAGIC 
# MAGIC Earlier, we did some exploratory work to explain why we should consider store-item level forecasting, arguing that aggregate level evaluation of the data hides some of the variability we might wish to action within our businesses.  Let's now generate an aggregate-level forecast for each product/item and allocate that forecast based on historical sales of those products within the different stores.  This is very typical of how retailers have overcome computational challenges in the past, aggregating sales across stores in a given market or region, and then allocating the results back out under the assumption these stores have similar temporal behaviors.
# MAGIC 
# MAGIC As before, we need to define a function to perform our forecasting work:

# COMMAND ----------

# DBTITLE 1,Define Aggregate Training & Forecasting Function
import pandas as pd

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

from fbprophet import Prophet

# train model & generate forecast from incoming dataset
def get_forecast_aggregate(item_pd, days_to_forecast):
  
  # retrieve item from dataframe
  item = item_pd['item'].iloc[0]
  
  # configure model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # fit to dataset
  model.fit( 
    item_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']] 
    )
  
  # make forecast
  future_item_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=True
    )
  
  # retrieve forecast
  forecast_item_pd = model.predict( future_item_pd )
  
  # assemple result set 
  forecast_item_pd['item']=item # assign item field
  
  # return forecast
  return forecast_item_pd[ 
      ['item', 'ds', 'trend', 'yhat_lower', 'yhat_upper',
       'trend_lower', 'trend_upper', 'multiplicative_terms',
       'multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',
       'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
       'yearly_upper', 'additive_terms', 'additive_terms_lower',
       'additive_terms_upper', 'yhat']
       ]

# COMMAND ----------

# DBTITLE 1,Train & Forecast in Aggregate
from pyspark.sql.functions import sum

# aggregate historical dataset across stores
agg_history_pd = (
  history
  .groupBy('item','date')
  .agg(sum('sales').alias('sales'))
  .toPandas()
  )

# retrieve distinct items for which to produce forecasts
items =(
  history
    .select('item')
    .distinct()
    .collect()
  )

# generate an aggeregate forecast per item
for i, item_ in enumerate(items):
  
  # get item
  item = item_['item']
  
  # extract item subset
  item_history_pd = agg_history_pd[ 
    (agg_history_pd['item']==item) 
    ].dropna()
  
  # generate forecast
  item_forecast_pd = get_forecast_aggregate(item_history_pd, days_to_forecast=30)
  
  # concatonate forecasts to build a single resultset 
  if i>0:
    item_accum_pd = item_accum_pd.append(item_forecast_pd, sort=False)
  else:
    item_accum_pd = item_forecast_pd

# display some results on screen
item_accum_pd.head()

# COMMAND ----------

# MAGIC %md Now let's the ratio of store-specific sales to total sales for each product.  This will give us the ratios that we need for allocation:

# COMMAND ----------

# DBTITLE 1,Calculate Allocation Ratios
from pyspark.sql.functions import expr, sum
from pyspark.sql.window import Window

# calculate per-item store sales ratios
item_store_ratios = (
  history
    .groupBy('store','item')
    .agg(sum('sales').alias('store_item_sales'))
    .withColumn('item_sales', sum('store_item_sales').over(Window.partitionBy('item')))
    .withColumn('ratio', expr('store_item_sales/item_sales'))
    .select('item', 'store', 'ratio', 'store_item_sales', 'item_sales')
    .orderBy(['item', 'store'])
  )

item_store_ratios_pd = item_store_ratios.toPandas()

display(item_store_ratios_pd)

# COMMAND ----------

# MAGIC %md Applying these ratios to our cross-store aggregates, we can now produce an allocated forecast for each store-item combination:

# COMMAND ----------

# DBTITLE 1,Perform Allocations
# peform store-level allocations
allocated_pd = (
  item_accum_pd[['item','ds','yhat']]
    .merge(
      item_store_ratios_pd, 
      how='outer', 
      left_on='item', 
      right_on='item'
      )
    )

# calculate allocated predicted values
allocated_pd['yhat_alloc'] = allocated_pd['yhat']*allocated_pd['ratio']

allocated_pd.head()

# COMMAND ----------

# MAGIC %md Let's now compare our allocated and store-item generated forecasts.  To do this, let's create temp views from these objects:

# COMMAND ----------

spark.createDataFrame(allocated_pd).createOrReplaceTempView('allocated_forecast')
store_item_accum_spark.createOrReplaceTempView('store_item_forecast')

# COMMAND ----------

# MAGIC %md Now, here is our forecast for item 1 by store as derived through allocation:

# COMMAND ----------

# DBTITLE 1,Allocated Item 1 Forecast, Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   store,
# MAGIC   ds,
# MAGIC   yhat_alloc
# MAGIC FROM allocated_forecast
# MAGIC WHERE item=1 AND ds >= '2018-01-01'
# MAGIC ORDER BY ds, store

# COMMAND ----------

# DBTITLE 1,Store-Specific Item 1 Forecast
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   store,
# MAGIC   ds,
# MAGIC   yhat
# MAGIC FROM store_item_forecast
# MAGIC WHERE item=1 AND ds >= '2018-01-01'
# MAGIC ORDER BY store, ds

# COMMAND ----------

# MAGIC %md But what do these differences really amount to?  How much higher and how much lower are my forecasts when compared to one another? If we just compare our forecasts for the 30-day future period over which we are estimating demand, the differences add up differently by product and by store which leads us to the conclusion that the *goodness* or *badness* of one strategy over the other really depends on the stores and products we are considering, but in some instances there are some sizeable misses that can have implications for out-of-stock and overstock scenarios:

# COMMAND ----------

# DBTITLE 1,Forecast Gap, Allocated vs. Store-Item Specific
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.store,
# MAGIC   a.item,
# MAGIC   SUM(abs(b.yhat_alloc - a.yhat)) as miss
# MAGIC FROM store_item_forecast a
# MAGIC INNER JOIN allocated_forecast b
# MAGIC   ON  a.item=b.item AND
# MAGIC       a.store=b.store AND
# MAGIC       a.ds=b.ds
# MAGIC WHERE a.ds >= '2018-01-01'
# MAGIC GROUP BY a.store, a.item
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md All of this is not to say that store-item level forecasting is always the right strategy (just like we aren't stating that FB Prophet is the right forecasting tool for every scenario).  What we are saying is that we want the option to explore a variety of forecasting strategies (and techniques).  In the past, computational limitations prohibited this but those days are behind us.
