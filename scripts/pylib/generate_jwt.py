import base64
import hashlib
import os
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

from pylib.environment import get_rsa_key


def get_jwt():
    """
    https://docs.snowflake.com/en/developer-guide/sql-api/authenticating#generating-a-jwt-in-python
    Generate a valid JWT token from snowflake account name, user name, private key and private key passphrase.
    To use the ChatDemoDev account, download the rsa_key.p8 from 1Password
    """

    rsa_key = get_rsa_key()
    private_key = load_pem_private_key(
        rsa_key,
        None,
        default_backend(),
    )

    public_key_raw = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    sha256hash = hashlib.sha256()
    sha256hash.update(public_key_raw)
    public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode("utf-8")

    # Generate JWT payload
    qualified_username = (
        os.getenv("SNOWFLAKE_ACCOUNT", "") + "." + os.getenv("SNOWFLAKE_USER", "")
    )
    now = datetime.now(timezone.utc)
    lifetime = timedelta(days=1000)
    payload = {
        "iss": qualified_username + "." + public_key_fp,
        "sub": qualified_username,
        "iat": now,
        "exp": now + lifetime,
    }
    return jwt.encode(payload, key=private_key, algorithm="RS256")  # type: ignore
