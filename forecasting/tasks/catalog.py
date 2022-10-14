from rfc_forecasting.common import Task
from rfc_forecasting.pipelines.catalog import CatalogPipeline


class CatalogTask(Task):

    def launch(self):
        self.logger.info("Launching catalog task")
        pipeline = CatalogPipeline(spark = self.spark)
        pipeline.initialize_catalog()
        self.logger.info("Catalog task finished!")

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = CatalogTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
