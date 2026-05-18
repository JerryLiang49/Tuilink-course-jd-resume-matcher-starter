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

    # Similarity is the final score used for matching. It may come from exact,
    # alias, fuzzy string, or embedding similarity depending on match_type.
    similarity: float
    match_type: Literal["exact", "alias", "fuzzy", "embedding"]

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

    # Name/category/importance are copied directly from the JD datapoint so the
    # UI can explain exactly which requirement was missed.
    name: str
    category: CategoryEnum
    importance: ImportanceEnum

    # Optional JD requirement metadata. If yoe is present, the JD explicitly
    # asked for that many years; otherwise proficiency is the qualitative signal
    # extracted from the JD.
    yoe: Optional[float] = None
    proficiency: ProficiencyEnum

    # Evidence ids point to the JD ExtractorState.sentences list.
    referenced_sentence_ids: List[str]


class ExtraResumeSkill(BaseModel):
    """Resume skill that did not map to any JD skill."""

    # Extra skills are not "bad"; they are resume strengths that did not improve
    # JD coverage under the current matching threshold.
    name: str
    category: CategoryEnum

    # Resume-side capability metadata retained for display/debugging.
    yoe: Optional[float] = None
    proficiency: ProficiencyEnum

    # Evidence ids point to the resume ExtractorState.sentences list.
    referenced_sentence_ids: List[str]


class DirectionalMatchResult(BaseModel):
    """One directional matching pass between source and target skill sets.

    The design doc calls for two passes:
    ``jd_to_resume`` asks how well the resume covers JD requirements, while
    ``resume_to_jd`` asks how much of the resume is relevant to the JD. Keeping
    both summaries makes asymmetric cases visible instead of hiding them behind
    one aggregate score.
    """

    # Direction of this pass. The source side is matched greedily into the
    # target side.
    direction: Literal["jd_to_resume", "resume_to_jd"]

    # Raw counts for this directional pass.
    source_skill_count: int
    target_skill_count: int
    matched_skill_count: int

    # source_coverage = matched source skills / all source skills
    # target_coverage = matched target skills / all target skills
    # f1 balances those two directional coverage values.
    source_coverage: float
    target_coverage: float
    f1: float

    # Matched evidence for this pass. Names are still stored as JD/resume
    # because the UI needs to show both documents consistently.
    matched_skills: List[MatchedSkill]


class MatcherResult(BaseModel):
    """Final local matcher output.

    Precision answers: "how many resume skills are relevant to the JD?"
    Recall answers: "how many JD skills are covered by the resume?"
    F1 balances the two. Required/preferred recall make requirement coverage
    easier to inspect than one aggregate score.
    """

    # Similarity threshold used by matcher.skill_matcher. Raising it makes
    # matching stricter; lowering it allows looser fuzzy matches.
    threshold: float

    # Raw counts used to interpret precision/recall.
    jd_skill_count: int
    resume_skill_count: int
    matched_skill_count: int

    # Backward-compatible aliases for the jd_to_resume directional result:
    # precision = matched resume skills / all resume skills
    # recall = matched JD skills / all JD skills
    # f1 = harmonic mean of precision and recall
    precision: float
    recall: float
    f1: float

    # Requirement-specific recall splits out required and preferred JD skills.
    required_recall: float
    preferred_recall: float

    # Among matched skills, the share where resume YOE satisfies explicit JD YOE.
    yoe_satisfaction_rate: float

    # Explicit directional summaries required by the matcher design. The
    # top-level precision/recall/f1 fields mirror ``jd_to_resume`` for older
    # callers.
    jd_to_resume: DirectionalMatchResult
    resume_to_jd: DirectionalMatchResult

    # Average of both directional F1 scores.
    bidirectional_score: float

    # Whether embedding similarity was requested and remained available.
    embedding_enabled: bool = False
    embedding_threshold: Optional[float] = None

    # Overall score intentionally uses the bidirectional score so it reflects
    # both requirement coverage and resume relevance.
    overall_score: float

    # Detailed evidence for how the aggregate metrics were produced.
    matched_skills: List[MatchedSkill]
    missing_required_skills: List[MissingSkill]
    missing_preferred_skills: List[MissingSkill]
    missing_unknown_importance_skills: List[MissingSkill]
    extra_resume_skills: List[ExtraResumeSkill]
