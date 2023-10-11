from copy import deepcopy
import click
from mlflow.exceptions import MlflowException
from . import copy_run
from . import copy_utils
from . click_options import (
    opt_src_model,
    opt_dst_model,
    opt_src_version,
    opt_src_registry_uri,
    opt_dst_registry_uri,
    opt_dst_experiment_name,
    opt_copy_lineage_tags
)
from mlflow_export_import.common.source_tags import ExportTags
from mlflow_export_import.common.click_options import opt_verbose
from mlflow_export_import.common import utils, model_utils, dump_utils
from mlflow_export_import.common.mlflow_utils import MlflowTrackingUriTweak

_logger = utils.getLogger(__name__)


def copy(src_model_name,
        src_model_version,
        dst_model_name,
        dst_experiment_name = None,
        src_tracking_uri = None,
        dst_tracking_uri = None,
        src_registry_uri = None,
        dst_registry_uri = None,
        copy_lineage_tags = False,
        verbose = False
    ):
    """
    Copy model version to another model in same or other tracking server (workspace).
    """
    src_client = copy_utils.mk_client(src_tracking_uri, src_registry_uri)
    dst_client = copy_utils.mk_client(dst_tracking_uri, dst_registry_uri)

    src_uri = f"{src_model_name}/{src_model_version}"
    print(f"Copying model version '{src_uri}' to '{dst_model_name}'")
    if verbose:
        dump_utils.dump_mlflow_client(src_client, "SRC")
        dump_utils.dump_mlflow_client(dst_client, "DST")
    copy_utils.create_registered_model(dst_client,  dst_model_name)
    src_version = src_client.get_model_version(src_model_name, src_model_version)
    if verbose:
        model_utils.dump_model_version(src_version, "Source Model Version")
    dst_version = _copy_model_version(src_version, dst_model_name, dst_experiment_name, src_client, dst_client, copy_lineage_tags)
    if verbose:
        model_utils.dump_model_version(dst_version, "Destination Model Version")
    dst_uri = f"{dst_version.name}/{dst_version.version}"
    print(f"Copied model version '{src_uri}' to '{dst_uri}'")
    return src_version, dst_version


def _copy_model_version(src_version, dst_model_name, dst_experiment_name, src_client, dst_client, copy_lineage_tags=False):
    if dst_experiment_name:
        dst_run = copy_run._copy(src_version.run_id, dst_experiment_name, src_client, dst_client)
    else:
        dst_run = src_client.get_run(src_version.run_id)

    mlflow_model_name = copy_utils.get_model_name(src_version.source)
    source_uri = f"{dst_run.info.artifact_uri}/{mlflow_model_name}"
    if copy_lineage_tags:
        tags = _add_lineage_tags(src_version, dst_run, dst_model_name, src_client, dst_client)
    else:
        tags = src_version.tags

    with MlflowTrackingUriTweak(dst_client):
        dst_version = dst_client.create_model_version(
            name = dst_model_name,
            source = source_uri,
            run_id = dst_run.info.run_id,
            tags = tags,
            description = src_version.description
        )
    if not model_utils.is_unity_catalog_model(dst_version.name):
        dst_client.transition_model_version_stage(dst_version.name, dst_version.version, src_version.current_stage)

    try:
        for alias in src_version.aliases:
            dst_client.set_registered_model_alias(dst_version.name, alias, dst_version.version)
    except MlflowException as e: # Non-UC Databricks MLflow has for some reason removed OSS MLflow support for aliases
        print(f"ERROR: error_code: {e.error_code}. Exception: {e}")

    return dst_client.get_model_version(dst_version.name, dst_version.version)


def _add_lineage_tags(src_version, run, dst_model_name, src_client, dst_client):
    prefix = f"{ExportTags.PREFIX_ROOT}.src_run"
    if src_version.run_id != run.info.run_id:
        run = src_client.get_run(src_version.run_id)

    tags = deepcopy(src_version.tags)

    tags[f"{ExportTags.PREFIX_ROOT}.src_version.name"] =  src_version.name
    tags[f"{ExportTags.PREFIX_ROOT}.src_version.version"] =  src_version.version
    tags[f"{ExportTags.PREFIX_ROOT}.src_version.run_id"] =  src_version.run_id

    tags[f"{ExportTags.PREFIX_ROOT}.src_client.tracking_uri"] = src_client.tracking_uri
    tags[f"{ExportTags.PREFIX_ROOT}.mlflow_exim.dst_client.tracking_uri"] = dst_client.tracking_uri

    copy_utils.add_tag(run.data.tags, tags, "mlflow.databricks.workspaceURL", prefix)
    copy_utils.add_tag(run.data.tags, tags, "mlflow.databricks.webappURL", prefix)
    copy_utils.add_tag(run.data.tags, tags, "mlflow.databricks.workspaceID", prefix)
    copy_utils.add_tag(run.data.tags, tags, "mlflow.user", prefix)

    if model_utils.is_unity_catalog_model(dst_model_name): # NOTE: Databricks UC model version tags don't accept '."
        tags = { k.replace(".","_"):v for k,v in tags.items() }

    return tags


@click.command()
@opt_src_model
@opt_src_version
@opt_dst_model
@opt_src_registry_uri
@opt_dst_registry_uri
@opt_dst_experiment_name
@opt_copy_lineage_tags
@opt_verbose

def main(src_model, src_version, dst_model, src_registry_uri, dst_registry_uri, dst_experiment_name, copy_lineage_tags, verbose):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    def mk_tracking_uri(registry_uri):
        if not registry_uri:
            return None
        if registry_uri.startswith("databricks-uc"):
            return registry_uri.replace("databricks-uc","databricks")
        else:
            return registry_uri

    src_tracking_uri = mk_tracking_uri(src_registry_uri)
    dst_tracking_uri = mk_tracking_uri(dst_registry_uri)

    print("Effective MLflow URIs:")
    print("  src_tracking_uri:", src_tracking_uri)
    print("  dst_tracking_uri:", dst_tracking_uri)
    print("  src_registry_uri:", src_registry_uri)
    print("  dst_registry_uri:", dst_registry_uri)
    copy(
        src_model,
        src_version, dst_model,
        dst_experiment_name,
        src_tracking_uri,
        dst_tracking_uri,
        src_registry_uri,
        dst_registry_uri,
        copy_lineage_tags = copy_lineage_tags,
        verbose = verbose
    )


if __name__ == "__main__":
    main()
