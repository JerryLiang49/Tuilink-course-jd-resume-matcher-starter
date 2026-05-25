"""JSON compatibility helpers for notebook/debug output and LLM parsing."""

import json
import numpy as np


def parse_response_json(response) -> dict:
    """Parse JSON from LangChain/OpenAI responses across content formats.

    LangChain chat responses usually expose ``content`` as a string, but newer
    Responses API integrations can return a list of content blocks. Keeping this
    parser in ``utils`` avoids duplicating that compatibility logic across LLM
    nodes.
    """

    content = response.content
    if isinstance(content, str):
        raw_text = content
    elif isinstance(content, list):
        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        ]
        raw_text = "\n".join(text_blocks)
    else:
        raw_text = str(content)

    return json.loads(raw_text)


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
