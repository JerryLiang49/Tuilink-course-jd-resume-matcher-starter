"""Local matcher for extracted JD and resume skills."""

# The Keywords Extractor turns raw documents into structured ``Skill`` objects.
# This matcher compares those objects in both directions: JD requirements against
# resume evidence, and resume evidence back against JD relevance. It can use
# OpenAI embeddings as a semantic similarity signal, but keeps deterministic
# string matching as a fallback for local tests or environments without an API
# key.

import os
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from models.datapoints import DataPoints, ImportanceEnum, Skill
from models.matcher_result import (
    DirectionalMatchResult,
    ExtraResumeSkill,
    MatchedSkill,
    MatcherResult,
    MissingSkill,
)


# Canonical aliases collapse common abbreviations and product-family names before
# scoring. Keep this list intentionally small: embeddings handle open-ended
# semantic synonyms, while aliases are reserved for abbreviations and product
# names where exact canonicalization is unambiguous.
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

DEFAULT_MATCH_USE_EMBEDDINGS = (
    os.getenv("MATCH_USE_EMBEDDINGS", "false").strip().lower()
    in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
)

try:
    DEFAULT_EMBEDDING_THRESHOLD = float(
        os.getenv("MATCH_EMBEDDING_THRESHOLD", "0.72").strip()
    )
except ValueError:
    DEFAULT_EMBEDDING_THRESHOLD = 0.72

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
    return {token for token in re.split(r"[\s/.-]+", value) if token}


def _deterministic_similarity(left_name: str, right_name: str) -> tuple[float, str]:
    """Return deterministic skill-name similarity and match type."""

    # Compare canonicalized names first. This catches exact matches and alias
    # matches before falling back to fuzzier scoring.
    left_normalized = _normalize_skill_name(left_name)
    right_normalized = _normalize_skill_name(right_name)

    if left_normalized == right_normalized:
        # If the raw names differ but canonical aliases match, mark it as alias.
        match_type = (
            "exact"
            if left_name.strip().lower() == right_name.strip().lower()
            else "alias"
        )
        return 1.0, match_type

    # If either side cannot produce usable tokens, the matcher cannot defend a
    # fuzzy match. Return no similarity instead of guessing.
    left_tokens = _tokenize(left_normalized)
    right_tokens = _tokenize(right_normalized)

    if not left_tokens or not right_tokens:
        return 0.0, "fuzzy"

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union

    # Containment is useful for cases like "data analysis" vs "analysis" or
    # "python programming" vs "python".
    # It is weighted below 1.0 so full exact/alias matches still rank higher.
    containment = intersection / min(len(left_tokens), len(right_tokens))

    # SequenceMatcher catches minor spelling or inflection differences, but it
    # is intentionally downweighted because strings can look similar while
    # meaning different technologies.
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()

    # Sequence similarity is useful for singular/plural or wording variants,
    # but keep its weight below token overlap so "Java" does not accidentally
    # match "JavaScript" at the default threshold.
    score = max(jaccard, containment * 0.95, sequence * 0.75)
    return _round_score(score), "fuzzy"


def _embedding_similarity(
    left_name: str,
    right_name: str,
    embedding_cache: dict[str, Any],
    embedding_state: dict[str, bool],
) -> float | None:
    """Return cosine similarity from embeddings, or None when unavailable."""

    if not embedding_state.get("enabled", False):
        return None

    try:
        # Lazy import keeps deterministic matcher tests usable without requiring
        # OPENAI_API_KEY at module import time.
        from utils.llm import LLM_USE_CACHE, get_embedding

        def get_cached_embedding(name: str) -> np.ndarray:
            normalized = _normalize_skill_name(name)
            if normalized not in embedding_cache:
                embedding_cache[normalized] = get_embedding(
                    normalized,
                    use_cache=LLM_USE_CACHE,
                )
            return embedding_cache[normalized]

        left_embedding = get_cached_embedding(left_name)
        right_embedding = get_cached_embedding(right_name)
        denominator = np.linalg.norm(left_embedding) * np.linalg.norm(right_embedding)
        if denominator == 0:
            return None

        return _round_score(float(np.dot(left_embedding, right_embedding) / denominator))
    except Exception:
        # If embeddings are unavailable locally, fall back to deterministic
        # matching instead of failing the whole matcher.
        embedding_state["enabled"] = False
        return None


def _similarity(
    left_name: str,
    right_name: str,
    *,
    embedding_cache: dict[str, Any],
    embedding_state: dict[str, bool],
    embedding_threshold: float,
) -> tuple[float, str]:
    """Return the best available similarity score and match type."""

    deterministic_score, deterministic_type = _deterministic_similarity(
        left_name,
        right_name,
    )
    embedding_score = _embedding_similarity(
        left_name,
        right_name,
        embedding_cache,
        embedding_state,
    )

    if (
        embedding_score is not None
        and embedding_score >= embedding_threshold
        and embedding_score > deterministic_score
    ):
        return embedding_score, "embedding"

    return deterministic_score, deterministic_type


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


def _best_target_match(
    source_skill: Skill,
    target_skills: list[Skill],
    used_target_indexes: set[int],
    threshold: float,
    *,
    embedding_cache: dict[str, Any],
    embedding_state: dict[str, bool],
    embedding_threshold: float,
) -> tuple[int | None, float, str]:
    """Find the best unused target skill for one source skill."""

    # A target skill can only satisfy one source skill. This keeps precision/recall
    # interpretable and avoids one broad resume skill inflating coverage.
    best_index: int | None = None
    best_score = 0.0
    best_match_type = "fuzzy"

    for index, target_skill in enumerate(target_skills):
        if index in used_target_indexes:
            continue

        # Hard skills and soft skills are not interchangeable for matching.
        if source_skill.category != target_skill.category:
            continue

        # Similarity only compares names. Category filtering already happened,
        # and YOE/proficiency are evaluated after the match is selected.
        score, match_type = _similarity(
            source_skill.name,
            target_skill.name,
            embedding_cache=embedding_cache,
            embedding_state=embedding_state,
            embedding_threshold=embedding_threshold,
        )
        if score > best_score:
            best_index = index
            best_score = score
            best_match_type = match_type

    if best_index is None or best_score < threshold:
        # Below threshold means "not covered" even if it is the best available
        # resume skill.
        return None, 0.0, "fuzzy"

    return best_index, best_score, best_match_type


def _directional_match(
    source_skills: list[Skill],
    target_skills: list[Skill],
    *,
    direction: str,
    source_is_jd: bool,
    threshold: float,
    embedding_cache: dict[str, Any],
    embedding_state: dict[str, bool],
    embedding_threshold: float,
) -> tuple[DirectionalMatchResult, list[Skill], set[int]]:
    """Run one greedy directional matching pass."""

    used_target_indexes: set[int] = set()
    matched_skills: list[MatchedSkill] = []
    unmatched_source_skills: list[Skill] = []

    for source_skill in source_skills:
        target_index, score, match_type = _best_target_match(
            source_skill,
            target_skills,
            used_target_indexes,
            threshold,
            embedding_cache=embedding_cache,
            embedding_state=embedding_state,
            embedding_threshold=embedding_threshold,
        )
        if target_index is None:
            unmatched_source_skills.append(source_skill)
            continue

        used_target_indexes.add(target_index)
        target_skill = target_skills[target_index]

        # MatchedSkill remains JD/resume-shaped even when the source direction is
        # resume_to_jd, because clients display evidence from both documents.
        jd_skill = source_skill if source_is_jd else target_skill
        resume_skill = target_skill if source_is_jd else source_skill
        matched_skills.append(_matched_skill(jd_skill, resume_skill, score, match_type))

    source_coverage = _safe_ratio(len(matched_skills), len(source_skills))
    target_coverage = _safe_ratio(len(matched_skills), len(target_skills))

    return (
        DirectionalMatchResult(
            direction=direction,  # type: ignore[arg-type]
            source_skill_count=len(source_skills),
            target_skill_count=len(target_skills),
            matched_skill_count=len(matched_skills),
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            f1=_f1(source_coverage, target_coverage),
            matched_skills=matched_skills,
        ),
        unmatched_source_skills,
        used_target_indexes,
    )


def match_resume_to_jd(
    jd_datapoints: DataPoints,
    resume_datapoints: DataPoints,
    *,
    threshold: float = 0.5,
    use_embeddings: bool = DEFAULT_MATCH_USE_EMBEDDINGS,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
) -> MatcherResult:
    """Match JD and resume skills in both directions and compute metrics.

    ``jd_to_resume`` measures requirement coverage. ``resume_to_jd`` measures
    how much of the resume evidence is relevant to the JD. Embeddings are an
    optional semantic signal; deterministic matching remains available for
    repeatable local tests and environments without embedding access.
    """

    jd_skills = sorted(jd_datapoints.skills, key=_importance_key)
    resume_skills = sorted(
        resume_datapoints.skills,
        key=lambda skill: (skill.category.value, _normalize_skill_name(skill.name)),
    )

    # Share embedding cache across both directions so each unique skill name is
    # embedded at most once per matcher invocation.
    embedding_cache: dict[str, Any] = {}
    embedding_state = {"enabled": use_embeddings}

    jd_to_resume, missing_jd_source_skills, used_resume_indexes = _directional_match(
        jd_skills,
        resume_skills,
        direction="jd_to_resume",
        source_is_jd=True,
        threshold=threshold,
        embedding_cache=embedding_cache,
        embedding_state=embedding_state,
        embedding_threshold=embedding_threshold,
    )
    resume_to_jd, _, _ = _directional_match(
        resume_skills,
        jd_skills,
        direction="resume_to_jd",
        source_is_jd=False,
        threshold=threshold,
        embedding_cache=embedding_cache,
        embedding_state=embedding_state,
        embedding_threshold=embedding_threshold,
    )

    matched_skills = jd_to_resume.matched_skills
    missing_skills = [_missing_skill(skill) for skill in missing_jd_source_skills]

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

    precision = jd_to_resume.target_coverage
    recall = jd_to_resume.source_coverage
    f1 = jd_to_resume.f1

    # Experience satisfaction is evaluated only for matched skills. Unmatched
    # skills already hurt recall and should not be double-counted here.
    yoe_satisfied = sum(1 for skill in matched_skills if skill.yoe_satisfied)
    yoe_satisfaction_rate = _safe_ratio(yoe_satisfied, len(matched_skills))
    bidirectional_score = _round_score((jd_to_resume.f1 + resume_to_jd.f1) / 2)

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
        jd_to_resume=jd_to_resume,
        resume_to_jd=resume_to_jd,
        bidirectional_score=bidirectional_score,
        embedding_enabled=embedding_state["enabled"],
        embedding_threshold=embedding_threshold if use_embeddings else None,
        overall_score=bidirectional_score,
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
