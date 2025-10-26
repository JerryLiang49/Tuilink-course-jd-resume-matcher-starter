from typing import List

from models.base import BaseModel
from models.datapoints import DataPoints
from models.sentence import Sentence


class ExtractorState(BaseModel):
    # Core document context (shared across phases)
    document: str
    sentences: List[Sentence]
    datapoints: DataPoints

