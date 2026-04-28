import logging
import os

from dotenv import load_dotenv

from pylib.util import strip_whitespace

logger = logging.getLogger(__name__)


def load_env():
    """
    Loads .env.* files from the root directory
    """

    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # .env will be check in to github
    env_path = os.path.join(root_folder, ".env")
    load_dotenv(env_path)
    # .env.secret should keep secret on local machine
    env_secret_path = os.path.join(root_folder, ".env.secret")
    load_dotenv(env_secret_path)


def get_rsa_key() -> bytes:
    """
    RSA key is used to create a JWT for querying the Cortex Search REST API
    Based on the environment, this can come from many places
    """
    # orchestrator_directory = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)), "../orchestrator"
    # )
    # wrf
    orchestrator_directory = "/code/users/boyiliu"
    # environment variable
    env_private_key = os.getenv("RSA_PRIVATE_KEY")
    if env_private_key:
        return str.encode(env_private_key)
    # local development
    account_to_key = {
        "CORTEXSEARCHSUMMITDEMO": "rsa_key_summit.p8",
        "SNOWPILOT_TEST": "rsa_key_summit.p8",
        "AKB73862": "rsa_key_boliu.p8",
        "_DEFAULT": "rsa_key.p8",
    }
    pkey_path = account_to_key.get(os.environ.get("SNOWFLAKE_ACCOUNT", "_DEFAULT"))
    for local_private_key_path in [
        f"{orchestrator_directory}/{pkey_path}",
        f"{orchestrator_directory}/rsa_key.p8",
    ]:
        if os.path.exists(local_private_key_path):
            with open(local_private_key_path, "rb") as pem_in:
                return pem_in.read()
    # Kubernetes with Hashicorp Vault
    kubernetes_private_key_path = "/vault/secrets/cortex-chat-rsa-key"
    if os.path.exists(kubernetes_private_key_path):
        with open(kubernetes_private_key_path, "rb") as pem_in:
            return pem_in.read()
    # throw exception
    raise Exception("❌ No RSA key, will not be able create JWT")


@strip_whitespace
def get_snowflake_password():
    current_directory = os.getcwd()
    # environmental variable
    password = os.getenv("SNOWFLAKE_PASSWORD")
    if password:
        return password
    # local development
    local_password_file = f"{current_directory}/orchestrator/demo_account_password"
    if os.getenv("SNOWFLAKE_ACCOUNT") == "CORTEXSEARCHDEMO":
        local_password_file = f"{current_directory}/prod_demo_account_password"
    if os.path.exists(local_password_file):
        with open(local_password_file, "r") as password_file:
            return password_file.read()
    # Kubernetes with Hashicorp vault
    kubernetes_password_file_path = "/vault/secrets/cortex-chat-account-password"
    if os.path.exists(kubernetes_password_file_path):
        with open(kubernetes_password_file_path, "r") as password_file:
            return password_file.read()
    # throw exception
    raise Exception("❌ No SNOWFLAKE_PASSWORD, will not be able to create session")


@strip_whitespace
def get_huggingface_token():
    """
    Gets huggingface token from secret file or env
    """
    # environment variable from .env.secret (local development)
    env_huggingface_token = os.getenv("HUGGINGFACE_READ_ONLY_TOKEN")
    if env_huggingface_token:
        return env_huggingface_token
    # Kubernetes with Hashicorp vault
    kubernetes_huggingface_token_file_path = (
        "/vault/secrets/cortex-chat-huggingface-token"
    )
    if os.path.exists(kubernetes_huggingface_token_file_path):
        with open(
            kubernetes_huggingface_token_file_path, "r"
        ) as huggingface_token_file:
            return huggingface_token_file.read()
    logger.warning(
        "⚠️ No HUGGINGFACE_READ_ONLY_TOKEN, will not be able to pull tokenizer from registry"
    )
    return ""


def get_sanitized_env():
    """
    Get sanitized environment variables (i.e. a dict of everything which isn't sensitive)
    """
    sanitized_env = dict(os.environ)
    # Create a list of keys to remove to avoid changing dictionary size during iteration
    keys_to_remove = [
        key
        for key in sanitized_env
        if "key" in key.lower() or "password" in key.lower()
    ]
    # Remove the keys
    for key in keys_to_remove:
        sanitized_env.pop(key)
    # Keep entries with TRULENS, SNOWFLAKE, EVAL, and OPENAI
    sanitized_env = {
        k: v
        for k, v in sanitized_env.items()
        if any(
            x in k
            for x in ["TRULENS", "SNOWFLAKE", "EVAL", "OPENAI", "CORTEX", "AZURE"]
        )
    }
    return sanitized_env
