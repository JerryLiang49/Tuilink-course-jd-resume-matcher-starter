"""JSON compatibility helpers for notebook/debug output."""

import numpy as np


def to_json_compatible(obj):
    """
    Recursively convert numpy arrays and numpy scalar types to Python native types
    for JSON serialization.

    This is useful for local notebook/debug output. Worker results that go to
    DynamoDB use ``convert_floats_to_decimal`` instead because DynamoDB has
    stricter number requirements than plain JSON.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_compatible(v) for v in obj]
    else:
        return obj
