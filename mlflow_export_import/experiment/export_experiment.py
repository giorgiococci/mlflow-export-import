"""
Exports an experiment to a directory.
"""

import os
import click
from dotenv import load_dotenv

from mlflow_export_import.common.click_options import (
    opt_experiment,
    opt_output_dir,
    opt_notebook_formats,
    opt_export_permissions,
    opt_run_start_time,
    opt_export_deleted_runs
)
from mlflow_export_import.common.iterators import SearchRunsIterator
from mlflow_export_import.common import utils, io_utils, mlflow_utils
from mlflow_export_import.common import ws_permissions_utils
from mlflow_export_import.common.timestamp_utils import fmt_ts_millis, utc_str_to_millis
from mlflow_export_import.client.client_utils import create_mlflow_client, create_mlflow_client_from_tracking_uri, create_dbx_client
from mlflow_export_import.run.export_run import export_run

_logger = utils.getLogger(__name__)


def export_experiment(
        experiment_id_or_name,
        output_dir,
        run_ids = None,
        export_permissions = False,
        run_start_time = None,
        export_deleted_runs = False,
        notebook_formats = None,
        mlflow_client = None,
        mlflow_tracking_uri = None
    ):
    """
    :param: experiment_id_or_name: Experiment ID or name.
    :param: output_dir: Output directory.
    :param: run_ids: List of run IDs to export. If None export then all run IDs.
    :param: export_permissions - Export Databricks permissions.
    :param: run_start_time - Only export runs started after this UTC time (inclusive). Format: YYYY-MM-DD.
    :param: notebook_formats: List of notebook formats to export. Values are SOURCE, HTML, JUPYTER or DBC.
    :param: mlflow_client: MLflow client.
    :return: Number of successful and number of failed runs.
    """
    if mlflow_tracking_uri:
        mlflow_client = mlflow_client or create_mlflow_client_from_tracking_uri()
    else:
        mlflow_client = mlflow_client or create_mlflow_client()
    #dbx_client = create_dbx_client(mlflow_client)

    run_start_time_str = run_start_time
    if run_start_time:
        run_start_time = utc_str_to_millis(run_start_time)

    exp = mlflow_utils.get_experiment(mlflow_client, experiment_id_or_name)
    msg = { "name": exp.name, "id": exp.experiment_id,
        "mlflow.experimentType": exp.tags.get("mlflow.experimentType", None),
        "lifecycle_stage": exp.lifecycle_stage
    }
    _logger.info(f"Exporting experiment: {msg}")
    ok_run_ids = []
    failed_run_ids = []
    num_runs_exported = 0
    if run_ids:
        for j,run_id in enumerate(run_ids):
            run = mlflow_client.get_run(run_id)
            _export_run(mlflow_client, run, output_dir, ok_run_ids, failed_run_ids,
                run_start_time, run_start_time_str, export_deleted_runs, notebook_formats)
            num_runs_exported += 1
    else:
        kwargs = {}
        if run_start_time:
            kwargs["filter"] = f"start_time > {run_start_time}"
        if export_deleted_runs:
            from mlflow.entities import ViewType
            kwargs["view_type"] = ViewType.ALL
        for j,run in enumerate(SearchRunsIterator(mlflow_client, exp.experiment_id, **kwargs)):
            _export_run(mlflow_client, run, output_dir, ok_run_ids, failed_run_ids,
                run_start_time, run_start_time_str, export_deleted_runs, notebook_formats)
            num_runs_exported += 1

    info_attr = {
        "num_total_runs": (num_runs_exported),
        "num_ok_runs": len(ok_run_ids),
        "num_failed_runs": len(failed_run_ids),
        "failed_runs": failed_run_ids
    }
    exp_dct = utils.strip_underscores(exp)
    exp_dct["_creation_time"] = fmt_ts_millis(exp.creation_time)
    exp_dct["_last_update_time"] = fmt_ts_millis(exp.last_update_time)
    exp_dct["tags"] = dict(sorted(exp_dct["tags"].items()))

    mlflow_attr = { "experiment": exp_dct , "runs": ok_run_ids }
    #if export_permissions:
    #    mlflow_attr["permissions"] = ws_permissions_utils.get_experiment_permissions(dbx_client, exp.experiment_id)
    io_utils.write_export_file(output_dir, "experiment.json", __file__, mlflow_attr, info_attr)

    msg = f"for experiment '{exp.name}' (ID: {exp.experiment_id})"
    if num_runs_exported==0:
        _logger.warning(f"No runs exported {msg}")
    elif len(failed_run_ids) == 0:
        _logger.info(f"{len(ok_run_ids)} runs succesfully exported {msg}")
    else:
        _logger.info(f"{len(ok_run_ids)}/{j} runs succesfully exported {msg}")
        _logger.info(f"{len(failed_run_ids)}/{j} runs failed {msg}")
    return len(ok_run_ids), len(failed_run_ids)


def _export_run(mlflow_client, run, output_dir, ok_run_ids, failed_run_ids,
        run_start_time, run_start_time_str,
        export_deleted_runs, notebook_formats
    ):
    if run_start_time and run.info.start_time < run_start_time:
        msg = {
            "run_id": {run.info.run_id},
            "experiment_id": {run.info.experiment_id},
            "start_time": fmt_ts_millis(run.info.start_time),
            "run_start_time": run_start_time_str
        }
        _logger.info(f"Not exporting run: {msg}")
        return

    is_success = export_run(
        run_id = run.info.run_id,
        output_dir = os.path.join(output_dir, run.info.run_id),
        export_deleted_runs = export_deleted_runs,
        notebook_formats = notebook_formats,
        mlflow_client = mlflow_client
    )
    if is_success:
        ok_run_ids.append(run.info.run_id)
    else:
        failed_run_ids.append(run.info.run_id)


@click.command()
@opt_experiment
@opt_output_dir
@opt_export_permissions
@opt_run_start_time
@opt_export_deleted_runs
@opt_notebook_formats

def main(experiment, output_dir, export_permissions, run_start_time, export_deleted_runs, notebook_formats):
    _logger.info("Options:")
    for k,v in locals().items():
        _logger.info(f"  {k}: {v}")
    export_experiment(
        experiment_id_or_name = experiment,
        output_dir = output_dir,
        export_permissions = export_permissions,
        run_start_time = run_start_time,
        export_deleted_runs = export_deleted_runs,
        notebook_formats = utils.string_to_list(notebook_formats),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
    )


if __name__ == "__main__":
    load_dotenv()
    
    opt_experiment="dev_taxi_fare_train_dev"
    opt_output_dir = "output"
    
    main()
