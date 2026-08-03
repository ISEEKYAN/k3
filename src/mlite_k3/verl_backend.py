"""VERL backend bridge that registers K3 before MLite runtime resolution."""

from mlite_k3 import register_model

register_model()

# Import the stock backend only after the external model is registered.  VERL
# imports this module independently in every Ray worker process.
from verl_mlite.engine import mlite_engine  # noqa: E402

__all__ = ["mlite_engine"]
