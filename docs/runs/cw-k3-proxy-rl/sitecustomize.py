"""Register the external K3 package in Python workers when explicitly enabled."""

import os


if os.environ.get("MLITE_K3_AUTO_REGISTER") == "1":
    from mlite_k3 import register_model

    register_model()
