import logging
import mlflow.pyfunc
import mlflow
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pathlib import Path
from prophet import Prophet
import time

logger = logging.getLogger(name = __name__)

class ForecastStoreItemModel(mlflow.pyfunc.PythonModel):
  """
  Class to train and use our Forecasting Models for Store/Item combinations
  """
    
  def __init__(
    self,
    experiment_id,
    model_path = "model",
    dst_path = "/dbfs/tmp/forecast/model",
    time_between_calls: int = 3,
    schema = "ds date, item int, y float, yhat float, yhat_upper float, yhat_lower float"
  ):
    """Will load all store/item logged models from the specified experiment_id"""

    experiment = mlflow.get_experiment(experiment_id)
    runs = mlflow.search_runs(experiment_ids = [experiment_id])
    self._filtered_runs = runs[runs["tags.mlflow.runName"].str.match(r"(run_item_[0-9]+_store_[0-9]+)")]
    self._model_path = model_path
    self._model_paths = {}
    self._dst_path = dst_path
    self._time_between_calls = time_between_calls
    self._schema = schema

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """

    logger.info("Loading context for Forecasting Model...")

      
  def make_prediction(
    self,
    model: Prophet,
    periods = 90,
    freq = 'd',
    include_history = True
  ) -> pd.DataFrame:
    
    future_pd = model.make_future_dataframe(
      periods=periods,
      freq=freq,
      include_history=include_history)

    return model.predict(future_pd)
  

  def predict(self, context, model_input):
    
    """This is an abstract function.
      Args:
          model_input: input dataframe containing sales data.
      Returns:
          model_output: dataframe containing forecast columns.
    """
    
    # Try loading model from DBFS, if it's not there, 
    # fetch from MLflow
    
    @pandas_udf(self._schema, PandasUDFType.GROUPED_MAP)
    def forecast_store_item(history_pd: pd.DataFrame) -> pd.DataFrame:

      model = None
      store = history_pd["store"].iloc[0]
      item = history_pd["item"].iloc[0]
      run_name = f"run_item_{item}_store_{store}"
      run_id = self._filtered_runs.loc[self._filtered_runs["run_name"] == run_name, "run_id"]
      final_dst_path = f"{self._dst_path}/{run_name}"

      if Path(final_dst_path).exists():
        logger.info(f"Using cached model from {final_dst_path}")
        model = mlflow.prophet.load_model(
          model_uri = self._model_paths[run_name]
        )
      else:
        logger.info(f"{final_dst_path} doesn't exist, fecthing from Mlflow...")
        time.sleep(self._time_between_calls)
        Path(final_dst_path).mkdir(parents=True, exist_ok=True)
        model = mlflow.prophet.load_model(
          model_uri = f"runs:/{run_id}/{self._model_path}",
          dst_path = final_dst_path
        )

      store = history_pd["store"].iloc[0]
      item = store = history_pd["item"].iloc[0]

      #predict forecast for the next days
      model_output = make_prediction(model)
      # --------------------------------------

      # ASSEMBLE EXPECTED RESULT SET
      # --------------------------------------
      # get relevant fields from forecast
      model_output['y'] = model_input['y']
      # get store & item from incoming data set
      model_output['store'] = store
      model_output['item'] = item
      # --------------------------------------

      # return expected dataset
      return model_output[['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]
    
    result = (
      model_input
        .groupBy('store', 'item')
          .apply(forecast_udf)
        .withColumn('training_date', f.current_date() )
        .withColumnRenamed('ds','date')
        .withColumnRenamed('y','sales')
        .withColumnRenamed('yhat','forecast')
        .withColumnRenamed('yhat_upper','forecast_upper')
        .withColumnRenamed('yhat_lower','forecast_lower')
    )
    
    return result
    
    
      