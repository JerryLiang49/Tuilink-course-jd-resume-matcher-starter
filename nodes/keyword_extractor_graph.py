"""LangGraph construction helpers for the JD/resume extraction pipeline."""

import logging
from langgraph.graph import StateGraph, END
from rexpand_pyutils_file import write_file

from models.extractor_state import ExtractorState
from models.datapoints import DataPoints
from nodes.preprocessor import preprocessor
from nodes.keyword_extractor import (
    comprehensive_checker,
    keywords_extractor,
    keywords_merger,
    modifier,
    supplementary_extractor,
    validator,
)


def _route_after_comprehensive_check(state: ExtractorState) -> str:
    """Route Phase 1 to supplementary extraction or finish."""

    state = ExtractorState.model_validate(state)
    if state.comprehensive_check and state.comprehensive_check.is_comprehensive:
        return END
    if state.phase1_iteration >= state.max_phase1_iterations:
        return END
    return "supplementary_extractor"


def _route_after_validation(state: ExtractorState) -> str:
    """Route Phase 2 to cleanup or finish."""

    state = ExtractorState.model_validate(state)
    if state.validation_result and state.validation_result.is_valid:
        return END
    if state.phase2_iteration >= state.max_phase2_iterations:
        return END
    return "modifier"


def build_phase1_graph() -> StateGraph:
    """Construct the LangGraph for Phase 1 with looping until comprehensive or max_iter."""

    # Phase 1 converts raw text into sentences, extracts skills, merges all
    # extraction passes, and loops through supplementary extraction until the
    # checker says the skill set is comprehensive.
    # StateGraph enforces that every node accepts and returns ExtractorState-like
    # data. LangGraph serializes state between nodes, so model field names are
    # part of the workflow contract.
    workflow = StateGraph(ExtractorState)

    # The preprocessor asks the LLM to split the document into meaningful
    # sentences and writes the result back into state.sentences.
    workflow.add_node("preprocessor", preprocessor)
    workflow.add_node("keywords_extractor", keywords_extractor)
    workflow.add_node("keywords_merger", keywords_merger)
    workflow.add_node("comprehensive_checker", comprehensive_checker)
    workflow.add_node("supplementary_extractor", supplementary_extractor)

    # The main path is preprocessor -> extractor -> merger -> checker. If the
    # checker finds missing skills, supplementary_extractor creates another
    # extraction pass and the graph merges/checks again.
    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", "keywords_extractor")
    workflow.add_edge("keywords_extractor", "keywords_merger")
    workflow.add_edge("keywords_merger", "comprehensive_checker")
    workflow.add_conditional_edges("comprehensive_checker", _route_after_comprehensive_check)
    workflow.add_edge("supplementary_extractor", "keywords_merger")

    return workflow


def build_phase2_graph() -> StateGraph:
    """Construct the validation/cleanup graph for extracted keywords."""

    workflow = StateGraph(ExtractorState)

    workflow.add_node("validator", validator)
    workflow.add_node("modifier", modifier)

    workflow.set_entry_point("validator")
    workflow.add_conditional_edges("validator", _route_after_validation)
    workflow.add_edge("modifier", "validator")

    return workflow


def run_phase1(document_text: str, max_iter: int = 3) -> ExtractorState:
    """Run Phase 1 end-to-end and return the merged, comprehensive ExtractorState."""

    workflow = build_phase1_graph()
    app = workflow.compile()

    # Start with the raw document, no sentences, and no extracted skills. The
    # graph fills sentences, extraction_history, datapoints, and the latest
    # comprehensive_check.
    initial = ExtractorState(
        document=document_text,
        datapoints=DataPoints(skills=[]),
        max_phase1_iterations=max_iter,
    )

    # LangGraph returns a dict-like state, so validate it back into our Pydantic
    # model before handing it to callers.
    result = app.invoke(initial)
    final_state = ExtractorState.model_validate(result)

    return final_state


def run_phase2(state: ExtractorState, max_iter: int = 3) -> ExtractorState:
    """Run deterministic validation/cleanup for extracted keywords."""

    state.phase = "phase2"
    state.max_phase2_iterations = max_iter

    workflow = build_phase2_graph()
    app = workflow.compile()

    result = app.invoke(state)
    return ExtractorState.model_validate(result)


def run_keywords_extractor(
    document_text: str,
    max_phase1_iter: int = 3,
    max_phase2_iter: int = 3,
) -> ExtractorState:
    """Run the local Keywords Extractor flow from raw text to validated skills."""

    phase1_state = run_phase1(document_text, max_iter=max_phase1_iter)
    return run_phase2(phase1_state, max_iter=max_phase2_iter)
