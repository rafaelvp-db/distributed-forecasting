# Databricks notebook source

import re

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

username = (sc._jvm.com.databricks.logging.AttributionContext.current().tags().get(
  sc._jvm.com.databricks.logging.BaseTagDefinitions.TAG_USER()).x())

userhome = f"dbfs:/user/{username}"

database = f"""{re.sub("[^a-zA-Z0-9]", "_", username)}_demand_db"""

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

spark.sql(f"USE {database}");
