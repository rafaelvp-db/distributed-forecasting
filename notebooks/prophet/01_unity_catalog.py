# Databricks notebook source
spark.sql("SHOW CATALOGS").filter("catalog = 'hackathon'").count()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC --- create a new catalog
# MAGIC CREATE CATALOG IF NOT EXISTS hackathon;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC --- Grant create & usage permissions to all users on the account
# MAGIC --- This also works for other account-level groups and individual users
# MAGIC 
# MAGIC GRANT CREATE, USAGE
# MAGIC ON CATALOG hackathon
# MAGIC TO `account users`;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC --- Check grants on the quickstart catalog
# MAGIC SHOW GRANT ON CATALOG hackathon;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC --- Create a new schema in the quick_start catalog
# MAGIC 
# MAGIC USE CATALOG hackathon;
# MAGIC CREATE SCHEMA IF NOT EXISTS sales
# MAGIC COMMENT "A new Unity Catalog schema called hackathon.sales";

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC -- Show schemas in the selected catalog
# MAGIC SHOW SCHEMAS;
