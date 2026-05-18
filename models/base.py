"""Shared Pydantic base model for project data contracts."""

from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """Project-wide Pydantic base model.

    All workflow models inherit from this class so they share the same string
    rendering behavior. The prompts often include model instances directly, and
    Pydantic's ``repr`` format is more explicit than the default object string.
    """

    def __str__(self):
        """Render models with field names and values for prompt/debug output."""
        return self.__repr__()
