from rfc_forecasting.tasks.catalog import CatalogTask
from pyspark.sql import SparkSession
from pathlib import Path
import logging

def test_create_catalog(spark: SparkSession):
    logging.info("Testing the Unity Catalog task")
    common_config = {"catalog_name": "hackathon", "schema_name": "sales"}
    test_catalog_config = {"output": common_config}
    catalog_task = CatalogTask()
    catalog_task.launch()
    
    test = spark.sql(f"SELECT CATALOG {common_config['catalog_name']}")
    assert test

