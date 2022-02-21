
from mlflow_export_import.model.import_model import _extract_model_path

run_id = "48cf29167ddb4e098da780f0959fb4cf"
model_path = "models/my_model"

def test_extract_model_path_databricks():
    source = f"dbfs:/databricks/mlflow-tracking/4072937019901104/{run_id}/artifacts/{model_path}"
    _test_extract_model_path(source)

def test_extract_model_path_oss():
    source = f"/opt/mlflow_server/local_mlrun/mlruns/3/{run_id}/artifacts/{model_path}"
    _test_extract_model_path(source)


def _test_extract_model_path(source):
    model_path2 = _extract_model_path(source, run_id)
    assert model_path == model_path2