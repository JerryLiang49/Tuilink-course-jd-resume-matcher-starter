from typing import List
from pydantic import ConfigDict

from models.base import BaseModel


class Sentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | int
    sentence: str


class RawSentences(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "required": ["sentences"]
    })
    sentences: List[str]
