from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    def __str__(self):
        return self.__repr__()
