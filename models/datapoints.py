from enum import Enum
from typing import Literal, Optional, List
from pydantic import ConfigDict

from models.base import BaseModel


class ImportanceEnum(str, Enum):
    UNKOWN = "unknown"
    REQUIRED = "required"
    PREFERRED = "preferred"


class CategoryEnum(str, Enum):
    SOFT_SKILLS = "soft_skills"
    HARD_SKILLS = "hard_skills"


class ProficiencyEnum(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    PROFICIENT = "proficient"
    ADVANCED = "advanced"
    EXPERT = "expert"


class BaseDataPoint(BaseModel):
    category: CategoryEnum
    importance: ImportanceEnum
    referenced_sentence_ids: List[str]


class Skill(BaseDataPoint):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": [
                "category",
                "importance",
                "referenced_sentence_ids",
                "name",
                "yoe",
                "proficiency",
            ]
        },
    )
    category: Literal[CategoryEnum.SOFT_SKILLS, CategoryEnum.HARD_SKILLS]
    name: str
    yoe: Optional[float] = None  # years of experience
    proficiency: ProficiencyEnum

    def get_yoe(self) -> float:
        """Get years of experience, converting from proficiency if yoe is None."""
        if self.yoe is not None:
            return self.yoe

        # Convert proficiency to years of experience using lower bounds of ranges
        proficiency_to_yoe = {
            ProficiencyEnum.BEGINNER: 0.0,  # 0-1 years
            ProficiencyEnum.INTERMEDIATE: 1.0,  # 1-2 years
            ProficiencyEnum.PROFICIENT: 2.0,  # 2-4 years
            ProficiencyEnum.ADVANCED: 4.0,  # 4-7 years
            ProficiencyEnum.EXPERT: 7.0,  # 7+ years
        }

        return proficiency_to_yoe[self.proficiency]


class DataPoints(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"required": ["skills"]}
    )
    skills: List[Skill]

