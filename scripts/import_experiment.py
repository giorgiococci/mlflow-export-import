import os
from dotenv import load_dotenv
import sys
MLFLOW_EXPORT_IMPORT_PATH= "/mnt/batch/tasks/shared/LS_root/mounts/clusters/vmprdwe0-2nw4m6/code/Users/caivano.luca/mlflow-export-import"
sys.path.insert(0, MLFLOW_EXPORT_IMPORT_PATH)
from mlflow_export_import.client import client_utils
from mlflow_export_import.common import mlflow_utils


from mlflow_export_import.experiment.import_experiment import import_experiment

load_dotenv()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI_IMPORT")

print(mlflow_tracking_uri)
#mlflow_tracking_uri = "azureml://250cb599-16b3-472a-b7a7-af5a3d5ade11.workspace.westeurope.api.azureml.ms/mlflow/v2.0/subscriptions/753a0b42-95dc-4871-b53e-160ceb0e6bc1/resourceGroups/rg-s-race-aml-dev-we/providers/Microsoft.MachineLearningServices/workspaces/amlsraceamldevwe01?" #dev

experiment_id_or_name="heart-condition-classifier-imported-3"
input_dir="output/8497eabd-226b-4a15-9d1f-95e65e141069"

import_experiment(
    experiment_name=experiment_id_or_name,
    input_dir=input_dir,
    import_source_tags = False,
    import_permissions = False,
    use_src_user_id = False,
    dst_notebook_dir = None,
    mlflow_tracking_uri=mlflow_tracking_uri
)