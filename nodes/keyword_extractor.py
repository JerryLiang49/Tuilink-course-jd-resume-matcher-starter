"""Keywords Extractor nodes for local JD/resume skill extraction.

The graph has two layers:

1. Phase 1 asks the LLM to extract skills, merges all extraction passes, and
   checks whether the result is comprehensive enough.
2. Phase 2 validates the merged skills and applies deterministic cleanup so the
   matcher receives normalized, evidence-backed datapoints.
"""

import json
import re
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from models.datapoints import (
    CategoryEnum,
    DataPoints,
    ImportanceEnum,
    ProficiencyEnum,
    Skill,
)
from models.extractor_state import (
    ComprehensiveCheckResult,
    ExtractorState,
    ValidationIssue,
    ValidationResult,
)
from utils.json import parse_response_json
from utils.llm import LLM_USE_CACHE, invoke_llm


def _sentences_as_prompt(state: ExtractorState) -> str:
    """Render sentence ids and text for LLM prompts."""

    return "\n".join(
        f"- id={sentence.id}: {sentence.sentence}" for sentence in state.sentences
    )


def _valid_sentence_ids(state: ExtractorState) -> set[str]:
    """Return sentence ids as strings because LLM JSON uses string ids."""

    return {str(sentence.id) for sentence in state.sentences}


def _sentence_id_sort_key(value: str) -> tuple[int, int | str]:
    """Sort numeric sentence ids numerically and other ids lexicographically."""

    return (0, int(value)) if value.isdigit() else (1, value)


def _normalize_name(name: str) -> str:
    """Normalize skill names for dedupe while preserving display names elsewhere."""

    return re.sub(r"\s+", " ", name.strip().lower())


def _importance_rank(value: ImportanceEnum) -> int:
    """Rank importance labels so merges can keep the strongest signal."""

    return {
        ImportanceEnum.UNKOWN: 0,
        ImportanceEnum.PREFERRED: 1,
        ImportanceEnum.REQUIRED: 2,
    }[value]


def _proficiency_rank(value: ProficiencyEnum) -> int:
    """Rank proficiency labels from weakest to strongest."""

    return {
        ProficiencyEnum.BEGINNER: 0,
        ProficiencyEnum.INTERMEDIATE: 1,
        ProficiencyEnum.PROFICIENT: 2,
        ProficiencyEnum.ADVANCED: 3,
        ProficiencyEnum.EXPERT: 4,
    }[value]


def _clean_skill(skill: Skill, valid_ids: set[str]) -> Skill | None:
    """Drop invalid evidence ids and reject skills without usable evidence."""

    referenced_sentence_ids = sorted(
        {
            str(sentence_id)
            for sentence_id in skill.referenced_sentence_ids
            if str(sentence_id) in valid_ids
        },
        key=_sentence_id_sort_key,
    )

    if not skill.name.strip() or not referenced_sentence_ids:
        return None

    yoe = skill.yoe
    if yoe is not None and yoe < 0:
        yoe = None

    return skill.model_copy(
        update={
            "name": re.sub(r"\s+", " ", skill.name.strip()),
            "referenced_sentence_ids": referenced_sentence_ids,
            "yoe": yoe,
        }
    )


def _merge_two_skills(existing: Skill, incoming: Skill) -> Skill:
    """Merge duplicate skills while preserving strongest metadata and evidence."""

    referenced_sentence_ids = sorted(
        set(existing.referenced_sentence_ids) | set(incoming.referenced_sentence_ids),
        key=_sentence_id_sort_key,
    )

    importance = (
        incoming.importance
        if _importance_rank(incoming.importance) > _importance_rank(existing.importance)
        else existing.importance
    )
    proficiency = (
        incoming.proficiency
        if _proficiency_rank(incoming.proficiency)
        > _proficiency_rank(existing.proficiency)
        else existing.proficiency
    )

    if existing.yoe is None:
        yoe = incoming.yoe
    elif incoming.yoe is None:
        yoe = existing.yoe
    else:
        yoe = max(existing.yoe, incoming.yoe)

    return existing.model_copy(
        update={
            "importance": importance,
            "proficiency": proficiency,
            "yoe": yoe,
            "referenced_sentence_ids": referenced_sentence_ids,
        }
    )


def merge_datapoints(datapoints_list: Iterable[DataPoints], state: ExtractorState) -> DataPoints:
    """Merge many extraction passes into one deterministic skill list."""

    valid_ids = _valid_sentence_ids(state)
    merged: dict[tuple[str, CategoryEnum], Skill] = {}

    for datapoints in datapoints_list:
        for raw_skill in datapoints.skills:
            skill = _clean_skill(raw_skill, valid_ids)
            if skill is None:
                continue

            key = (_normalize_name(skill.name), skill.category)
            if key in merged:
                merged[key] = _merge_two_skills(merged[key], skill)
            else:
                merged[key] = skill

    return DataPoints(
        skills=sorted(
            merged.values(),
            key=lambda skill: (_normalize_name(skill.name), skill.category.value),
        )
    )


def keywords_extractor(state: ExtractorState) -> ExtractorState:
    """Extract initial skill datapoints from preprocessed sentences."""

    state.phase = "phase1"
    state.step = "phase1:keywords_extractor"
    if state.phase1_iteration == 0:
        state.phase1_iteration = 1

    system_prompt = """
    You extract structured skills from job descriptions or resumes.
    Return only skills explicitly supported by the provided sentence ids.
    Use hard_skills for tools, technologies, methods, domains, credentials, and
    measurable professional capabilities. Use soft_skills for communication,
    leadership, collaboration, ownership, and similar behavioral skills.
    For job descriptions, mark must-have requirements as required and nice-to-
    have requirements as preferred. For resumes, use unknown unless the text
    clearly states the skill is a target requirement.
    If years of experience are explicitly stated, include yoe. Otherwise infer
    the closest proficiency label from the evidence.
    """

    user_prompt = f"""
    Extract the important skills from these sentences.

    Sentences:
    {_sentences_as_prompt(state)}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={
            "format": {
                "name": "datapoints",
                "strict": True,
                "type": "json_schema",
                "schema": DataPoints.model_json_schema(),
            }
        },
        use_cache=LLM_USE_CACHE,
    )

    datapoints = DataPoints.model_validate(parse_response_json(response))
    state.record_extraction_pass(
        "keywords_extractor",
        datapoints,
        reason="Initial skill extraction from preprocessed sentences.",
    )
    return state


def keywords_merger(state: ExtractorState) -> ExtractorState:
    """Merge all keyword extraction passes into state.datapoints."""

    state.step = "phase1:keywords_merger"
    state.datapoints = merge_datapoints(
        (extraction.datapoints for extraction in state.extraction_history),
        state,
    )
    return state


def comprehensive_checker(state: ExtractorState) -> ExtractorState:
    """Check whether Phase 1 skills cover the meaningful source sentences."""

    state.step = "phase1:comprehensive_checker"

    system_prompt = """
    You check whether extracted skills comprehensively cover a job description
    or resume. Be strict about missing important technical skills, required
    qualifications, tools, seniority signals, credentials, and important soft
    skills. Do not require extraction of irrelevant headers, company boilerplate,
    dates, or generic filler sentences.
    """

    user_prompt = f"""
    Sentences:
    {_sentences_as_prompt(state)}

    Current extracted skills:
    {state.datapoints.model_dump_json(indent=2)}

    Decide if the extracted skills are comprehensive. If not, list missing
    keywords/concepts and the sentence ids that need another extraction pass.
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={
            "format": {
                "name": "comprehensive_check",
                "strict": True,
                "type": "json_schema",
                "schema": ComprehensiveCheckResult.model_json_schema(),
            }
        },
        use_cache=LLM_USE_CACHE,
    )

    result = ComprehensiveCheckResult.model_validate(parse_response_json(response))

    valid_ids = _valid_sentence_ids(state)
    result.missing_sentence_ids = [
        str(sentence_id)
        for sentence_id in result.missing_sentence_ids
        if str(sentence_id) in valid_ids
    ]

    # If the loop hits its configured limit, stop Phase 1 with the best merged
    # datapoints instead of looping forever.
    if state.phase1_iteration >= state.max_phase1_iterations:
        result.is_comprehensive = True
        if result.reason:
            result.reason += " Stopping because max_phase1_iterations was reached."
        else:
            result.reason = "Stopping because max_phase1_iterations was reached."

    state.comprehensive_check = result
    if result.is_comprehensive:
        state.phase = "phase2"
    return state


def supplementary_extractor(state: ExtractorState) -> ExtractorState:
    """Extract missing skills requested by the comprehensive checker."""

    state.step = "phase1:supplementary_extractor"
    state.phase1_iteration += 1

    missing_sentence_ids = set()
    missing_keywords: list[str] = []
    if state.comprehensive_check:
        missing_sentence_ids = {
            str(sentence_id)
            for sentence_id in state.comprehensive_check.missing_sentence_ids
        }
        missing_keywords = state.comprehensive_check.missing_keywords

    target_sentences = [
        sentence
        for sentence in state.sentences
        if not missing_sentence_ids or str(sentence.id) in missing_sentence_ids
    ]
    sentence_text = "\n".join(
        f"- id={sentence.id}: {sentence.sentence}" for sentence in target_sentences
    )

    system_prompt = """
    You are the supplementary skill extractor. Add only missing skills that are
    not already represented in the current extracted skills. Every skill must
    cite at least one provided sentence id.
    """

    user_prompt = f"""
    Current extracted skills:
    {state.datapoints.model_dump_json(indent=2)}

    Missing concepts to focus on:
    {json.dumps(missing_keywords, ensure_ascii=False)}

    Candidate sentences:
    {sentence_text}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={
            "format": {
                "name": "supplementary_datapoints",
                "strict": True,
                "type": "json_schema",
                "schema": DataPoints.model_json_schema(),
            }
        },
        use_cache=LLM_USE_CACHE,
    )

    datapoints = DataPoints.model_validate(parse_response_json(response))
    state.record_extraction_pass(
        "supplementary_extractor",
        datapoints,
        reason="Supplementary extraction for comprehensive checker gaps.",
    )
    return state


def validator(state: ExtractorState) -> ExtractorState:
    """Validate merged skills before they are handed to the matcher."""

    state.phase = "phase2"
    state.step = "phase2:validator"

    valid_ids = _valid_sentence_ids(state)
    issues: list[ValidationIssue] = []
    seen_keys: set[tuple[str, str]] = set()

    if not state.datapoints.skills:
        issues.append(
            ValidationIssue(
                issue_id="no_skills",
                severity="error",
                message="No skills were extracted from the document.",
                suggested_fix="Run extraction again or inspect the preprocessed sentences.",
            )
        )

    for index, skill in enumerate(state.datapoints.skills):
        skill_key = (_normalize_name(skill.name), skill.category.value)
        if skill_key in seen_keys:
            issues.append(
                ValidationIssue(
                    issue_id=f"duplicate_skill_{index}",
                    severity="error",
                    message=f"Duplicate skill found: {skill.name}",
                    skill_names=[skill.name],
                    suggested_fix="Merge duplicate skills into one datapoint.",
                )
            )
        seen_keys.add(skill_key)

        invalid_refs = [
            str(sentence_id)
            for sentence_id in skill.referenced_sentence_ids
            if str(sentence_id) not in valid_ids
        ]
        if invalid_refs:
            issues.append(
                ValidationIssue(
                    issue_id=f"invalid_refs_{index}",
                    severity="error",
                    message=f"Skill {skill.name} references invalid sentence ids.",
                    skill_names=[skill.name],
                    referenced_sentence_ids=invalid_refs,
                    suggested_fix="Remove invalid sentence references.",
                )
            )

        if not skill.referenced_sentence_ids:
            issues.append(
                ValidationIssue(
                    issue_id=f"missing_refs_{index}",
                    severity="error",
                    message=f"Skill {skill.name} has no sentence evidence.",
                    skill_names=[skill.name],
                    suggested_fix="Drop the skill or attach a valid evidence sentence.",
                )
            )

        if skill.yoe is not None and skill.yoe < 0:
            issues.append(
                ValidationIssue(
                    issue_id=f"negative_yoe_{index}",
                    severity="error",
                    message=f"Skill {skill.name} has negative years of experience.",
                    skill_names=[skill.name],
                    suggested_fix="Clear yoe or replace it with a non-negative value.",
                )
            )

    state.validation_result = ValidationResult(
        is_valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        confidence=1.0,
        reason=(
            "Deterministic validation passed."
            if not issues
            else "Deterministic validation found cleanup issues."
        ),
    )
    if state.validation_result.is_valid:
        state.phase = "complete"
        state.step = "complete"
    return state


def modifier(state: ExtractorState) -> ExtractorState:
    """Apply deterministic cleanup for validator issues."""

    state.phase = "phase2"
    state.step = "phase2:modifier"
    state.phase2_iteration += 1

    cleaned = merge_datapoints([state.datapoints], state)
    state.record_extraction_pass(
        "modifier",
        cleaned,
        reason="Deterministic cleanup after validation.",
    )
    return state
