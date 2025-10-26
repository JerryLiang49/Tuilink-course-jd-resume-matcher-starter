import json
from typing import Any
from decimal import Decimal
import numpy as np


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal objects from DynamoDB."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to int if it's a whole number, otherwise to float
            if obj % 1 == 0:
                return int(obj)
            else:
                return float(obj)
        return super(DecimalEncoder, self).default(obj)


def json_dumps_decimal(obj: Any) -> str:
    """JSON dumps with Decimal support."""
    return json.dumps(obj, cls=DecimalEncoder, default=str)


def convert_floats_to_decimal(obj):
    """Recursively convert numeric values (including numpy types) to Decimal for DynamoDB compatibility."""
    # Preserve Decimal as-is
    if isinstance(obj, Decimal):
        return obj
    # Avoid treating booleans as integers
    if isinstance(obj, bool):
        return obj
    # Handle numpy arrays explicitly (convert to list then recurse)
    if isinstance(obj, np.ndarray):
        return [convert_floats_to_decimal(item) for item in obj.tolist()]
    # Handle float and numpy floating types
    if isinstance(obj, (float, np.floating)):
        return Decimal(str(float(obj)))
    # Handle int and numpy integer types (but not bool)
    if isinstance(obj, (int, np.integer)):
        return Decimal(str(int(obj)))
    # Recurse into mappings
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    # Recurse into sequences (list/tuple/set)
    if isinstance(obj, (list, tuple, set)):
        return [convert_floats_to_decimal(item) for item in obj]
    # Fallback: leave object unchanged
    return obj
