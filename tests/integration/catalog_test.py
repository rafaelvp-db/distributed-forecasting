from rfc_forecasting.tasks.catalog import CatalogTask

def test_etl():
    common_config = {"catalog_name": "hackathon", "schema_name": "sales"}
    catalog_task = CatalogTask(init_conf=common_config)
    catalog_task.launch()
    _count = catalog_task\
        .spark.sql("SHOW CATALOGS")\
        .filter("catalog = 'hackathon'")\
        .count()
    assert _count == 1
