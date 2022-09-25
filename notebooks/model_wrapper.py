import mlflow.pyfunc


class ForecastingModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use our Forecasting Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """

        #TODO: code to load the models for each store/item combination
        #self.model = 

    def predict(self, context, model_input):
        """This is an abstract function.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        return self.model