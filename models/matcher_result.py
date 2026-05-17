"""Structured result models for local JD/resume matching."""

from typing import List, Literal, Optional

from models.base import BaseModel
from models.datapoints import CategoryEnum, ImportanceEnum, ProficiencyEnum


class MatchedSkill(BaseModel):
    """One JD skill matched to one resume skill."""

    # Skill names are kept separately because JD/resume wording may differ even
    # when the matcher considers them equivalent.
    jd_skill_name: str
    resume_skill_name: str

    # Category is copied from the JD skill and should match the resume skill
    # category because the matcher does not cross-match hard and soft skills.
    category: CategoryEnum

    # Importance comes from the JD side because it describes the requirement.
    jd_importance: ImportanceEnum

    # Similarity is the deterministic name similarity score used for matching.
    similarity: float
    match_type: Literal["exact", "alias", "fuzzy"]

    # Skill-level experience/proficiency comparison fields.
    jd_yoe: Optional[float] = None
    resume_yoe: Optional[float] = None
    yoe_gap: Optional[float] = None
    yoe_satisfied: bool = True
    jd_proficiency: ProficiencyEnum
    resume_proficiency: ProficiencyEnum

    # Evidence ids point back to each document's ExtractorState.sentences list.
    jd_sentence_ids: List[str]
    resume_sentence_ids: List[str]


class MissingSkill(BaseModel):
    """JD skill that was not covered by the resume."""

    name: str
    category: CategoryEnum
    importance: ImportanceEnum
    yoe: Optional[float] = None
    proficiency: ProficiencyEnum
    referenced_sentence_ids: List[str]


class ExtraResumeSkill(BaseModel):
    """Resume skill that did not map to any JD skill."""

    name: str
    category: CategoryEnum
    yoe: Optional[float] = None
    proficiency: ProficiencyEnum
    referenced_sentence_ids: List[str]


class MatcherResult(BaseModel):
    """Final local matcher output.

    Precision answers: "how many resume skills are relevant to the JD?"
    Recall answers: "how many JD skills are covered by the resume?"
    F1 balances the two. Required/preferred recall make requirement coverage
    easier to inspect than one aggregate score.
    """

    threshold: float
    jd_skill_count: int
    resume_skill_count: int
    matched_skill_count: int

    precision: float
    recall: float
    f1: float
    required_recall: float
    preferred_recall: float
    yoe_satisfaction_rate: float

    # Overall score intentionally uses F1 so it remains easy to explain.
    overall_score: float

    matched_skills: List[MatchedSkill]
    missing_required_skills: List[MissingSkill]
    missing_preferred_skills: List[MissingSkill]
    missing_unknown_importance_skills: List[MissingSkill]
    extra_resume_skills: List[ExtraResumeSkill]
