"""LangGraph construction helpers for the JD/resume extraction pipeline."""

import logging
from langgraph.graph import StateGraph, END
from rexpand_pyutils_file import write_file

from models.extractor_state import ExtractorState
from models.datapoints import DataPoints
from nodes.preprocessor import preprocessor


def build_phase1_graph() -> StateGraph:
    """Construct the Phase 1 extraction graph.

    Phase 1 currently has one responsibility: convert raw document text into a
    list of sentence objects. The graph abstraction is intentionally used even
    for this simple case so future phases can add more nodes, loops, and
    conditional edges without changing the worker handler contract.
    """

    # StateGraph enforces that every node accepts and returns ExtractorState-like
    # data. LangGraph serializes state between nodes, so model field names are
    # part of the workflow contract.
    workflow = StateGraph(ExtractorState)

    # The preprocessor asks the LLM to split the document into meaningful
    # sentences and writes the result back into state.sentences.
    workflow.add_node("preprocessor", preprocessor)

    # Current graph is linear: start at preprocessor, then finish.
    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", END)

    return workflow


def run_phase1(document_text: str, max_iter: int = 3) -> ExtractorState:
    """Run Phase 1 end-to-end and return the populated state.

    ``max_iter`` is kept in the signature for compatibility with the intended
    iterative extractor design, but it is not used by the current one-node graph.
    """

    workflow = build_phase1_graph()
    app = workflow.compile()

    # Start with the raw document, no sentences, and no extracted skills. The
    # preprocessor will fill sentences; later phases should fill datapoints.
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
