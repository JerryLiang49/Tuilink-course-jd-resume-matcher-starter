"""Deterministic local matcher for extracted JD and resume skills."""

# This module intentionally avoids LLM/embedding calls. The Keywords Extractor
# already turns raw documents into structured ``Skill`` objects, so the matcher
# can run locally by comparing normalized skill names, categories, and
# experience metadata. That makes matching cheap, repeatable, and easy to test
# in notebooks or Lambda.

import re
from difflib import SequenceMatcher

from models.datapoints import CategoryEnum, DataPoints, ImportanceEnum, Skill
from models.matcher_result import (
    ExtraResumeSkill,
    MatchedSkill,
    MatcherResult,
    MissingSkill,
)


# Canonical aliases collapse common abbreviations and product-family names before
# scoring. This lets obvious equivalents match exactly without an embedding
# model, e.g. "Py" -> "python" and "PostgreSQL" -> "sql".
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
    "rest api development": "rest api",
    "rest api": "rest api",
    "rest apis": "rest api",
    "restful api development": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "sql databases": "sql",
    "ts": "typescript",
}

# Required JD skills should get first access to resume skills. Because matching
# is one-to-one, sorting required requirements first prevents a preferred skill
# from consuming the best resume evidence for a required one.
IMPORTANCE_ORDER = {
    ImportanceEnum.REQUIRED: 0,
    ImportanceEnum.PREFERRED: 1,
    ImportanceEnum.UNKOWN: 2,
}


def _round_score(value: float) -> float:
    """Round scores consistently for API/notebook output."""

    # Four decimals are precise enough for UI/debug display and avoid noisy
    # floating-point tails in DynamoDB/API responses.
    return round(value, 4)


def _normalize_skill_name(name: str) -> str:
    """Normalize skill names before exact/fuzzy comparison."""

    # Normalize punctuation/spacing but keep characters meaningful for skills,
    # such as C++, C#, Node.js, CI/CD, and GPT-4.
    normalized = name.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9+#./-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Apply aliasing after text normalization so variants like "React.js" and
    # "react js" converge to the same canonical form.
    return ALIASES.get(normalized, normalized)


def _tokenize(value: str) -> set[str]:
    """Tokenize normalized skill names for overlap scoring."""

    # Token overlap handles compound skills where word order or separators vary,
    # such as "data-analysis" vs "data analysis".
    tokens: set[str] = set()
    for token in re.split(r"[\s/.-]+", value):
        if not token:
            continue
        # Normalize common plural forms after aliasing. This catches cases where
        # the extractor emits "REST APIs" on one side and "REST API development"
        # on the other without making the whole matcher depend on embeddings.
        if token == "apis":
            token = "api"
        tokens.add(token)

    return tokens


def _similarity(jd_name: str, resume_name: str) -> tuple[float, str]:
    """Return deterministic skill-name similarity and match type."""

    # Compare canonicalized names first. This catches exact matches and alias
    # matches before falling back to fuzzier scoring.
    jd_normalized = _normalize_skill_name(jd_name)
    resume_normalized = _normalize_skill_name(resume_name)

    if jd_normalized == resume_normalized:
        # If the raw names differ but canonical aliases match, mark it as alias.
        match_type = (
            "exact"
            if jd_name.strip().lower() == resume_name.strip().lower()
            else "alias"
        )
        return 1.0, match_type

    # If either side cannot produce usable tokens, the matcher cannot defend a
    # fuzzy match. Return no similarity instead of guessing.
    jd_tokens = _tokenize(jd_normalized)
    resume_tokens = _tokenize(resume_normalized)

    if not jd_tokens or not resume_tokens:
        return 0.0, "fuzzy"

    intersection = len(jd_tokens & resume_tokens)
    union = len(jd_tokens | resume_tokens)
    jaccard = intersection / union

    # Containment is useful for cases like "data analysis" vs "analysis" or
    # "python programming" vs "python".
    # It is weighted below 1.0 so full exact/alias matches still rank higher.
    containment = intersection / min(len(jd_tokens), len(resume_tokens))

    # SequenceMatcher catches minor spelling or inflection differences, but it
    # is intentionally downweighted because strings can look similar while
    # meaning different technologies.
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

    # Missing skills are copied from the JD side because they represent
    # requirements the resume failed to cover.
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

    # Extra resume skills are still useful: they may be strengths, but they did
    # not contribute to JD coverage under the current threshold.
    return ExtraResumeSkill(
        name=skill.name,
        category=skill.category,
        yoe=skill.yoe,
        proficiency=skill.proficiency,
        referenced_sentence_ids=[str(value) for value in skill.referenced_sentence_ids],
    )


def _matched_skill(
    jd_skill: Skill, resume_skill: Skill, score: float, match_type: str
) -> MatchedSkill:
    """Build the public matched-skill record."""

    # Only enforce YOE when the JD explicitly mentions YOE. If the JD has only a
    # qualitative proficiency label, the label remains visible but does not
    # block coverage.
    jd_yoe = jd_skill.yoe
    resume_yoe = resume_skill.get_yoe()

    # yoe_gap is positive when the candidate exceeds the JD requirement and
    # negative when they fall short. It is None when the JD did not specify a
    # numeric YOE requirement.
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

    # Empty denominator means there was nothing to satisfy. For example, a JD
    # with no preferred skills has preferred_recall = 1.0 by convention.
    if denominator == 0:
        return 1.0
    return _round_score(float(numerator) / float(denominator))


def _f1(precision: float, recall: float) -> float:
    """Compute rounded F1 from precision and recall."""

    # Avoid division by zero for the degenerate case where both precision and
    # recall are zero.
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

    # A resume skill can only satisfy one JD skill. This keeps precision/recall
    # interpretable and avoids one broad resume skill inflating coverage.
    best_index: int | None = None
    best_score = 0.0
    best_match_type = "fuzzy"

    for index, resume_skill in enumerate(resume_skills):
        if index in used_resume_indexes:
            continue

        # Hard skills and soft skills are not interchangeable for matching.
        if jd_skill.category != resume_skill.category:
            continue

        # Similarity only compares names. Category filtering already happened,
        # and YOE/proficiency are evaluated after the match is selected.
        score, match_type = _similarity(jd_skill.name, resume_skill.name)
        if score > best_score:
            best_index = index
            best_score = score
            best_match_type = match_type

    if best_index is None or best_score < threshold:
        # Below threshold means "not covered" even if it is the best available
        # resume skill.
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

    # Track used resume skills by index after sorting. This makes the matching
    # deterministic even when two resume skills have similar names.
    used_resume_indexes: set[int] = set()
    matched_skills: list[MatchedSkill] = []
    missing_skills: list[MissingSkill] = []

    # Greedy matching is enough for this local implementation because required
    # JD skills are sorted first and each skill picks the best currently unused
    # resume skill above threshold.
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

    # Requirement-specific recall is often more actionable than aggregate recall
    # because missing required skills is more important than missing preferred
    # skills.
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

    # Experience satisfaction is evaluated only for matched skills. Unmatched
    # skills already hurt recall and should not be double-counted here.
    yoe_satisfied = sum(1 for skill in matched_skills if skill.yoe_satisfied)
    yoe_satisfaction_rate = _safe_ratio(yoe_satisfied, len(matched_skills))

    # Split missing skills by JD importance so API clients can show required
    # gaps separately from preferred/unknown gaps.
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
