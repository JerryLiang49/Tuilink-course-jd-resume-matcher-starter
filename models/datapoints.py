"""Structured datapoint models extracted from resumes and job descriptions."""

from enum import Enum
from typing import Literal, Optional, List
from pydantic import ConfigDict

from models.base import BaseModel


class ImportanceEnum(str, Enum):
    """How strongly a skill is requested by the source document."""

    # Keep the existing misspelling for compatibility with existing prompts/data.
    UNKOWN = "unknown"
    REQUIRED = "required"
    PREFERRED = "preferred"


class CategoryEnum(str, Enum):
    """High-level skill category used by extraction and matching."""

    SOFT_SKILLS = "soft_skills"
    HARD_SKILLS = "hard_skills"


class ProficiencyEnum(str, Enum):
    """Normalized skill proficiency labels.

    These values let the workflow compare resume skills against JD skills even
    when no explicit years-of-experience value appears in the text.
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    PROFICIENT = "proficient"
    ADVANCED = "advanced"
    EXPERT = "expert"


class BaseDataPoint(BaseModel):
    """Common fields for extracted structured facts.

    A data point should always point back to the sentence ids that support it.
    That reference is important for explainability and for later validation.
    """

    # Soft skill vs hard skill. Subclasses can narrow this type further.
    category: CategoryEnum

    # Whether the source document says the skill is required, preferred, or
    # unknown. For resumes this can be unknown because resumes describe history
    # rather than job requirements.
    importance: ImportanceEnum

    # Sentence ids from ``ExtractorState.sentences`` used as evidence.
    referenced_sentence_ids: List[str]


class Skill(BaseDataPoint):
    """A single normalized skill extracted from a JD or resume."""

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

    # This model currently supports only skills, and each skill must be either a
    # hard skill or a soft skill.
    category: Literal[CategoryEnum.SOFT_SKILLS, CategoryEnum.HARD_SKILLS]

    # Canonical skill name, such as "Python", "SQL", or "communication".
    name: str

    # Explicit years of experience if the text provides it. This is optional
    # because many resumes/JDs describe proficiency qualitatively instead.
    yoe: Optional[float] = None  # years of experience

    # Normalized proficiency label used when explicit YOE is unavailable.
    proficiency: ProficiencyEnum

    def get_yoe(self) -> float:
        """Return comparable years of experience for scoring.

        If the extractor found an explicit numeric YOE, use it directly. If not,
        map the proficiency label to the lower bound of an approximate range so
        downstream matchers can compare skills on one numeric scale.
        """

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
    """Container for all structured facts extracted from one document."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"required": ["skills"]}
    )

    # Skill list extracted from either a resume or job description.
    skills: List[Skill]
