import mlflow

def load_model(self, model_name: str = "model_name", stage: str = "Production"):
    """
    Loads the trained model from the model registry and returns
    it as mlflow.pyfunc.PyFuncModel to be used to do the predictions.
    IMPORTANT: The function returns the MLflow model, not the predictions.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

    # Use the abstract function in FastTextWrapper to fetch the trained model.
    model = loaded_model.predict(pd.DataFrame([1, 2, 3]))

    self.logger.info("Type of the model is")
    self.logger.info(type(model))

    return model