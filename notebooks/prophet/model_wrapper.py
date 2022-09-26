import logging
import mlflow
import os
import pandas as pd
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
    time_between_calls: int = 0.5,
  ):
    """Will load all store/item logged models from the specified experiment_id"""

    logger.info(f"Experiment ID: {experiment_id}")
    self._experiment_id = experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    logger.info(f"Experiment: {experiment}")
    self._runs = mlflow.search_runs(experiment_ids = [experiment_id])
    self._model_path = model_path
    self._dst_path = dst_path
    self._time_between_calls = time_between_calls

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
    Args:
        context: MLflow context where the model artifact is stored.
    """

    logger.info("Loading context for Forecasting Model...")

  
  def predict(self, context, model_input):

    # Try loading model from DBFS, if it's not there, 
    # fetch from MLflow
    
    model = None
    future_df = model_input
    store = future_df["store"].iloc[0]
    item = future_df["item"].iloc[0]
    run_name = f"run_item_{item}_store_{store}"
    run_id = self._runs.loc[
      self._runs["tags.mlflow.runName"] == run_name, "run_id"
    ].values[0]
    
    time.sleep(self._time_between_calls)
    model = mlflow.prophet.load_model(model_uri = f"runs:/{run_id}/{self._model_path}")

    #predict forecast for the next days
    model_output = model.predict(future_df)
    # --------------------------------------

    # ASSEMBLE EXPECTED RESULT SET
    # --------------------------------------
    # get store & item from incoming data set
    model_output['store'] = store
    model_output['item'] = item
    # --------------------------------------

    # return expected dataset
    logger.info(f"Model output - columns: {model_output.columns}")
    return model_output[["ds", "store", "item", "yhat", "yhat_upper", "yhat_lower"]]


    
    
      