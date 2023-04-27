import os
import json
import requests
import click
from mlflow_export_import.common import utils
from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.client import USER_AGENT
from mlflow_export_import.client import mlflow_auth_utils

_logger = utils.getLogger(__name__)
_TIMEOUT = 15


class HttpClient():
    """ Wrapper for GET and POST methods for Databricks REST APIs  - standard Databricks API and MLflow API. """
    def __init__(self, api_name, host=None, token=None):
        """
        :param api_name: Name of base API such as 'api/2.0' or 'api/2.0/mlflow'.
        :param host: Host name of tracking server.
        :param token: Databricks token.
        """
        self.api_uri = "?"
        if host is None:
            (host, token) = mlflow_auth_utils.get_mlflow_host_token()
            if host is None:
                raise MlflowExportImportException(
                    "MLflow tracking URI (MLFLOW_TRACKING_URI environment variable) is not configured correctly",
                    http_status_code=401)
        self.api_uri = os.path.join(host, api_name)
        self.token = token

    def _get(self, resource, params=None):
        """ Executes an HTTP GET call
        :param resource: Relative path name of resource such as cluster/list
        :param params: Dict of query parameters 
        """
        uri = self._mk_uri(resource)
        rsp = requests.get(uri, headers=self._mk_headers(), json=params, timeout=_TIMEOUT)
        self._check_response(rsp, params)
        return rsp

    def get(self, resource, params=None):
        return json.loads(self._get(resource, params).text)


    def _post(self, resource, data=None):
        """ Executes an HTTP POST call
        :param resource: Relative path name of resource such as runs/search
        :param data: Request request payload
        """
        uri = self._mk_uri(resource)
        data = json.dumps(data) if data else None
        rsp = requests.post(uri, headers=self._mk_headers(), data=data, timeout=_TIMEOUT)
        self._check_response(rsp, data)
        return rsp

    def post(self, resource, data=None):
        return json.loads(self._post(resource, data).text)


    def _put(self, resource, data=None):
        """ Executes an HTTP PUT call
        :param resource: Relative path name of resource.
        :param data: Request payload
        """
        uri = self._mk_uri(resource)
        rsp = requests.put(uri, headers=self._mk_headers(), data=data, timeout=_TIMEOUT)
        self._check_response(rsp)
        return rsp

    def put(self, resource, data=None):
        return json.loads(self._put(resource, data).text)


    def _patch(self, resource, data=None):
        """ Executes an HTTP PATCH call
        :param resource: Relative path name of resource.
        :param data: Request payload
        """
        uri = self._mk_uri(resource)
        rsp = requests.patch(uri, headers=self._mk_headers(), data=data, timeout=_TIMEOUT)
        self._check_response(rsp)
        return rsp

    def patch(self, resource, data=None):
        return json.loads(self._patch(resource, data).text)


    def _delete(self, resource, data=None):
        """ Executes an HTTP POST call
        :param resource: Relative path name of resource such as runs/search
        :param data: Post request payload
        """
        uri = self._mk_uri(resource)
        data = json.dumps(data) if data else None
        rsp = requests.delete(uri, headers=self._mk_headers(), data=data, timeout=_TIMEOUT)
        self._check_response(rsp, data)
        return rsp

    def delete(self, resource, data=None):
        return json.loads(self._delete(resource, data).text)


    def _mk_headers(self):
        headers = { "User-Agent": USER_AGENT }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}" 
        return headers

    def _mk_uri(self, resource):
        return f"{self.api_uri}/{resource}"

    def _get_response_text(self, rsp):
        try:
            return rsp.json()
        except requests.exceptions.JSONDecodeError:
            return rsp.text

    def _check_response(self, rsp, params=None):
        if rsp.status_code < 200 or rsp.status_code > 299:
            raise MlflowExportImportException(
               rsp.reason,
               http_status_code = rsp.status_code,
               http_reason = rsp.reason,
               uri = rsp.url, 
               params = params,
               text = self._get_response_text(rsp)
            )

    def __repr__(self): 
        return self.api_uri


class DatabricksHttpClient(HttpClient):
    def __init__(self, host=None, token=None):
        super().__init__("api/2.0", host, token)


class MlflowHttpClient(HttpClient):
    def __init__(self, host=None, token=None):
        super().__init__("api/2.0/mlflow", host, token)


@click.command()
@click.option("--api", help="API: mlflow|databricks.", default="mlflow", type=str)
@click.option("--resource", help="API resource such as 'experiments/search'.", required=True, type=str)
@click.option("--method", help="HTTP method: GET|POST.", default="GET", type=str)
@click.option("--params", help="HTTP GET query parameters as JSON.", required=False, type=str)
@click.option("--data", help="HTTP POST data as JSON.", required=False, type=str)
@click.option("--output-file", help="Output file.", required=False, type=str)
@click.option("--verbose", help="Verbose.", type=bool, default=False, show_default=True)

def main(api, resource, method, params, data, output_file, verbose):
    def write_output(rsp, output_file):
        if output_file:
            _logger.info(f"Output file: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(rsp.text)
        else:
            _logger.info(rsp.text)

    if verbose:
        _logger.info("Options:")
        for k,v in locals().items():
            _logger.info(f"  {k}: {v}")

    client = DatabricksHttpClient() if api == "databricks" else MlflowHttpClient()
    method = method.upper() 
    if "GET" == method:
        if params:
            params = json.loads(params)
        rsp = client._get(resource, params)
        write_output(rsp, output_file)
    elif "POST" == method:
        rsp = client._post(resource, data)
        write_output(rsp, output_file)
    else:
        _logger.error(f"Unsupported HTTP method '{method}'")


if __name__ == "__main__":
    main()
