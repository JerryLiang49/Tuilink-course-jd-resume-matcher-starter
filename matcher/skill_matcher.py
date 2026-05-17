"""Deterministic local matcher for extracted JD and resume skills."""

import re
from difflib import SequenceMatcher

from models.datapoints import CategoryEnum, DataPoints, ImportanceEnum, Skill
from models.matcher_result import (
    ExtraResumeSkill,
    MatchedSkill,
    MatcherResult,
    MissingSkill,
)


ALIASES = {
    "ai": "artificial intelligence",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "ci cd": "ci/cd",
    "ci/cd pipelines": "ci/cd",
    "excel": "microsoft excel",
    "js": "javascript",
    "k8s": "kubernetes",
    "llms": "large language models",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "mysql": "sql",
    "postgres": "sql",
    "postgres sql": "sql",
    "postgresql": "sql",
    "py": "python",
    "react js": "react",
    "react.js": "react",
    "sql databases": "sql",
    "ts": "typescript",
}

IMPORTANCE_ORDER = {
    ImportanceEnum.REQUIRED: 0,
    ImportanceEnum.PREFERRED: 1,
    ImportanceEnum.UNKOWN: 2,
}


def _round_score(value: float) -> float:
    """Round scores consistently for API/notebook output."""

    return round(value, 4)


def _normalize_skill_name(name: str) -> str:
    """Normalize skill names before exact/fuzzy comparison."""

    normalized = name.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9+#./-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return ALIASES.get(normalized, normalized)


def _tokenize(value: str) -> set[str]:
    """Tokenize normalized skill names for overlap scoring."""

    return {token for token in re.split(r"[\s/.-]+", value) if token}


def _similarity(jd_name: str, resume_name: str) -> tuple[float, str]:
    """Return deterministic skill-name similarity and match type."""

    jd_normalized = _normalize_skill_name(jd_name)
    resume_normalized = _normalize_skill_name(resume_name)

    if jd_normalized == resume_normalized:
        # If the raw names differ but canonical aliases match, mark it as alias.
        match_type = "exact" if jd_name.strip().lower() == resume_name.strip().lower() else "alias"
        return 1.0, match_type

    jd_tokens = _tokenize(jd_normalized)
    resume_tokens = _tokenize(resume_normalized)

    if not jd_tokens or not resume_tokens:
        return 0.0, "fuzzy"

    intersection = len(jd_tokens & resume_tokens)
    union = len(jd_tokens | resume_tokens)
    jaccard = intersection / union

    # Containment is useful for cases like "data analysis" vs "analysis" or
    # "python programming" vs "python".
    containment = intersection / min(len(jd_tokens), len(resume_tokens))
    sequence = SequenceMatcher(None, jd_normalized, resume_normalized).ratio()

    # Sequence similarity is useful for singular/plural or wording variants,
    # but keep its weight below token overlap so "Java" does not accidentally
    # match "JavaScript" at the default threshold.
    score = max(jaccard, containment * 0.95, sequence * 0.75)
    return _round_score(score), "fuzzy"


def _importance_key(skill: Skill) -> tuple[int, str]:
    """Sort JD skills so required skills get first chance at resume matches."""

    return (IMPORTANCE_ORDER[skill.importance], _normalize_skill_name(skill.name))


def _missing_skill(skill: Skill) -> MissingSkill:
    """Convert a JD Skill into a missing-skill result item."""

    return MissingSkill(
        name=skill.name,
        category=skill.category,
        importance=skill.importance,
        yoe=skill.yoe,
        proficiency=skill.proficiency,
        referenced_sentence_ids=[str(value) for value in skill.referenced_sentence_ids],
    )


def _extra_resume_skill(skill: Skill) -> ExtraResumeSkill:
    """Convert an unmatched resume Skill into a result item."""

    return ExtraResumeSkill(
        name=skill.name,
        category=skill.category,
        yoe=skill.yoe,
        proficiency=skill.proficiency,
        referenced_sentence_ids=[str(value) for value in skill.referenced_sentence_ids],
    )


def _matched_skill(jd_skill: Skill, resume_skill: Skill, score: float, match_type: str) -> MatchedSkill:
    """Build the public matched-skill record."""

    # Only enforce YOE when the JD explicitly mentions YOE. If the JD has only a
    # qualitative proficiency label, the label remains visible but does not
    # block coverage.
    jd_yoe = jd_skill.yoe
    resume_yoe = resume_skill.get_yoe()
    yoe_gap = None if jd_yoe is None else _round_score(resume_yoe - jd_yoe)
    yoe_satisfied = True if jd_yoe is None else resume_yoe >= jd_yoe

    return MatchedSkill(
        jd_skill_name=jd_skill.name,
        resume_skill_name=resume_skill.name,
        category=jd_skill.category,
        jd_importance=jd_skill.importance,
        similarity=score,
        match_type=match_type,  # type: ignore[arg-type]
        jd_yoe=jd_yoe,
        resume_yoe=resume_yoe,
        yoe_gap=yoe_gap,
        yoe_satisfied=yoe_satisfied,
        jd_proficiency=jd_skill.proficiency,
        resume_proficiency=resume_skill.proficiency,
        jd_sentence_ids=[str(value) for value in jd_skill.referenced_sentence_ids],
        resume_sentence_ids=[str(value) for value in resume_skill.referenced_sentence_ids],
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio while treating empty denominators as perfect."""

    if denominator == 0:
        return 1.0
    return _round_score(float(numerator) / float(denominator))


def _f1(precision: float, recall: float) -> float:
    """Compute rounded F1 from precision and recall."""

    if precision + recall == 0:
        return 0.0
    return _round_score(2 * precision * recall / (precision + recall))


def _best_resume_match(
    jd_skill: Skill,
    resume_skills: list[Skill],
    used_resume_indexes: set[int],
    threshold: float,
) -> tuple[int | None, float, str]:
    """Find the best unused resume skill for one JD skill."""

    best_index: int | None = None
    best_score = 0.0
    best_match_type = "fuzzy"

    for index, resume_skill in enumerate(resume_skills):
        if index in used_resume_indexes:
            continue

        # Hard skills and soft skills are not interchangeable for matching.
        if jd_skill.category != resume_skill.category:
            continue

        score, match_type = _similarity(jd_skill.name, resume_skill.name)
        if score > best_score:
            best_index = index
            best_score = score
            best_match_type = match_type

    if best_index is None or best_score < threshold:
        return None, 0.0, "fuzzy"

    return best_index, best_score, best_match_type


def match_resume_to_jd(
    jd_datapoints: DataPoints,
    resume_datapoints: DataPoints,
    *,
    threshold: float = 0.5,
) -> MatcherResult:
    """Match resume skills against JD skills and compute local metrics.

    The matcher is intentionally deterministic. It does not call an LLM or an
    embedding API, so it is safe to run repeatedly in notebooks, tests, and AWS
    Lambda after the Keywords Extractor has produced structured skills.
    """

    jd_skills = sorted(jd_datapoints.skills, key=_importance_key)
    resume_skills = sorted(
        resume_datapoints.skills,
        key=lambda skill: (skill.category.value, _normalize_skill_name(skill.name)),
    )

    used_resume_indexes: set[int] = set()
    matched_skills: list[MatchedSkill] = []
    missing_skills: list[MissingSkill] = []

    for jd_skill in jd_skills:
        resume_index, score, match_type = _best_resume_match(
            jd_skill,
            resume_skills,
            used_resume_indexes,
            threshold,
        )
        if resume_index is None:
            missing_skills.append(_missing_skill(jd_skill))
            continue

        used_resume_indexes.add(resume_index)
        matched_skills.append(
            _matched_skill(jd_skill, resume_skills[resume_index], score, match_type)
        )

    extra_resume_skills = [
        _extra_resume_skill(skill)
        for index, skill in enumerate(resume_skills)
        if index not in used_resume_indexes
    ]

    required_total = sum(
        1 for skill in jd_skills if skill.importance == ImportanceEnum.REQUIRED
    )
    required_matched = sum(
        1 for skill in matched_skills if skill.jd_importance == ImportanceEnum.REQUIRED
    )
    preferred_total = sum(
        1 for skill in jd_skills if skill.importance == ImportanceEnum.PREFERRED
    )
    preferred_matched = sum(
        1 for skill in matched_skills if skill.jd_importance == ImportanceEnum.PREFERRED
    )

    precision = _safe_ratio(len(matched_skills), len(resume_skills))
    recall = _safe_ratio(len(matched_skills), len(jd_skills))
    f1 = _f1(precision, recall)

    yoe_satisfied = sum(1 for skill in matched_skills if skill.yoe_satisfied)
    yoe_satisfaction_rate = _safe_ratio(yoe_satisfied, len(matched_skills))

    return MatcherResult(
        threshold=threshold,
        jd_skill_count=len(jd_skills),
        resume_skill_count=len(resume_skills),
        matched_skill_count=len(matched_skills),
        precision=precision,
        recall=recall,
        f1=f1,
        required_recall=_safe_ratio(required_matched, required_total),
        preferred_recall=_safe_ratio(preferred_matched, preferred_total),
        yoe_satisfaction_rate=yoe_satisfaction_rate,
        overall_score=f1,
        matched_skills=matched_skills,
        missing_required_skills=[
            skill
            for skill in missing_skills
            if skill.importance == ImportanceEnum.REQUIRED
        ],
        missing_preferred_skills=[
            skill
            for skill in missing_skills
            if skill.importance == ImportanceEnum.PREFERRED
        ],
        missing_unknown_importance_skills=[
            skill
            for skill in missing_skills
            if skill.importance == ImportanceEnum.UNKOWN
        ],
        extra_resume_skills=extra_resume_skills,
    )
