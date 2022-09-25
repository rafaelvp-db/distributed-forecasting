import mlflow
from .model_wrapper import ForecastingModelWrapper

def save_model(
    df: pd.DataFrame,
    model_name: str = "model_name"
):
    """
    Trains and saves our model
    Args:
        df (pd.DataFrame): Input Dataframe
        model_name (str): Model will be saved with this name 
    Returns:
        trained_model: Returns the trained model
    """
    
    #TODO: code to train multiple models using Pandas UDF
    # and save resulting models
    
    with mlflow.start_run() as run:
        #TODO: log trained models
        #mlflow.pyfunc.log_model()

    return trained_model