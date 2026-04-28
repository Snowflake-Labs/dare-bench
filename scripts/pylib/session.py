import os
from typing import Any

from snowflake.connector import connect
from snowflake.snowpark import Session

from pylib.logger import get_logger
from pylib.environment import get_snowflake_password
from pylib.util import remove_http_scheme

logger = get_logger()


def create_session() -> Session:
    """
    Creates a snowpark session
    TODO (vv): deprecate this if not used
    """
    # oauth token path
    # context = get_orchestrator_context() # TODO (vv) replace context / deprecate
    context = None
    if (
        context["x_snowflake_oauth_token"]
        and context["x_snowflake_account"]
        and context["x_snowflake_url"]
    ):
        connect_params = {
            "account": context["x_snowflake_account"],
            "token": context["x_snowflake_oauth_token"],
            "authenticator": "oauth",
            "host": remove_http_scheme(context["x_snowflake_url"]),
        }
        return Session.builder.configs({**connect_params}).create()

    # TODO after summit: delete all the code below or relocate it to a separate function so it can continue to be used by eval tool

    base_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

    # otherwise, use the password to create the session
    password = get_snowflake_password()
    if password:
        connection_parameters = {
            **base_parameters,
            "host": os.getenv("SNOWFLAKE_HOST"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "password": password,
        }
        connection: Any = connect(**connection_parameters)
        session = Session.builder.configs({"connection": connection}).create()
        return session

    raise Exception("❌ Could not create session: no token or SNOWFLAKE_PASSWORD")


def create_eval_session() -> Session:
    """
    Create an eval session
    """
    base_parameters = {
        "account": os.getenv("EVAL_ACCOUNT"),
        "warehouse": os.getenv("EVAL_WAREHOUSE"),
        "database": os.getenv("EVAL_DATABASE"),
        "schema": os.getenv("EVAL_SCHEMA"),
    }
    password = None
    current_directory = os.getcwd() # /eval
    if current_directory.endswith("eval"):
        eval_password_file = os.path.join(current_directory, '..', 'secrets', 'eval_account_password')
    else: # from root /chat
        eval_password_file = os.path.join(current_directory, 'secrets', 'eval_account_password')
    if os.path.exists(eval_password_file):
        with open(eval_password_file, "r") as password_file:
            password = password_file.read()

    if password is not None:
        connection_parameters = {
            **base_parameters,
            "host": os.getenv("EVAL_HOST"),
            "user": os.getenv("EVAL_USER"),
            "role": os.getenv("EVAL_ROLE"),
            "password": password,
        }
        connection: Any = connect(**connection_parameters)
        session = Session.builder.configs({"connection": connection}).create()
        return session

    raise Exception("❌ Could not create eval session: no SNOWFLAKE_PASSWORD")
