"""
One-time setup script: stores the Lakebase connection URL as a Databricks
secret. Run this locally (with the Databricks CLI configured) or from a
notebook - never commit the resulting secret value anywhere.

OpenAlex requires no API key, so this is the only secret this project needs.

Usage:
    python setup_secrets.py
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase connection URL: "),
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)
