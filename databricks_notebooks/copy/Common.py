# Databricks notebook source
# Common - copy

# COMMAND ----------

# MAGIC %pip install git+https:///github.com/mlflow/mlflow-export-import@issue-138-copy-model-version#egg=mlflow-export-import
# MAGIC
# MAGIC # %pip install /dbfs/home/andre.mesarovic@databricks.com/lib/wheels/mlflow_export_import-1.2.0-py3-none-any.whl

# COMMAND ----------

import mlflow
print("mlflow.version:", mlflow.__version__)

# COMMAND ----------

from mlflow_export_import.copy.local_utils import dump_obj_as_json

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0: 
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

from mlflow.utils import databricks_utils
_host_name = databricks_utils.get_browser_hostname()
print("host_name:", _host_name)

def display_registered_model_version_uri(model_name, version):
    if _host_name:
        if "." in model_name: # is unity catalog model
            model_name = model_name.replace(".","/")
            uri = f"https://{_host_name}/explore/data/models/{model_name}/version/{version}"
        else:
            uri = f"https://{_host_name}/#mlflow/models/{model_name}/versions/{version}"
        displayHTML("""<b>Registered Model Version URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def copy_model_version(
        src_model_name,
        src_model_version,
        dst_model_name,
        dst_experiment_name, 
        dst_tracking_uri = "databricks",
        verbose = False 
    ):
    from mlflow_export_import.copy.copy_model_version import copy
    from mlflow_export_import.copy.local_utils import is_unity_catalog_model 
    
    def mk_registry_uri(model_name):
        return "databricks-uc" if is_unity_catalog_model(model_name) else "databricks"

    src_registry_uri = mk_registry_uri(src_model_name)
    dst_registry_uri = mk_registry_uri(dst_model_name)
    print("src_registry_uri:", src_registry_uri)
    print("dst_registry_uri:", dst_registry_uri)

    return copy(
        src_model_name,
        src_model_version,
        dst_model_name,
        dst_experiment_name, 
        src_tracking_uri = "databricks",  
        dst_tracking_uri = dst_tracking_uri,
        src_registry_uri = src_registry_uri, 
        dst_registry_uri = dst_registry_uri,
        verbose = verbose 
    )