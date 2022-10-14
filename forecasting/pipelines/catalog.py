

class CatalogPipeline:
    def __init__(self, spark):
        self.spark = spark

    def initialize_catalog(
        self,
        catalog_name = "hackathon",
        schema_name = "sales"
    ):

        self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name};")
        self.spark.sql(f"""
            GRANT CREATE, USAGE
            ON CATALOG {catalog_name}
            TO `account users`;
        """)
        self.spark.sql(f"USE CATALOG {catalog_name};")
        self.spark.sql(f"""CREATE SCHEMA IF NOT EXISTS {schema_name}
            COMMENT 'A new Unity Catalog schema called {catalog_name}.{schema_name}';
        """)

        



