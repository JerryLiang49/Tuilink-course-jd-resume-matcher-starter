import logging
from langgraph.graph import StateGraph, END
from rexpand_pyutils_file import write_file

from models.extractor_state import ExtractorState
from models.datapoints import DataPoints
from nodes.preprocessor import preprocessor


def build_phase1_graph() -> StateGraph:
    """Construct the LangGraph for Phase 1 with looping until comprehensive or max_iter."""
    workflow = StateGraph(ExtractorState)

    workflow.add_node("preprocessor", preprocessor)

    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", END)

    return workflow


def run_phase1(document_text: str, max_iter: int = 3) -> ExtractorState:
    """Run Phase 1 end-to-end and return the merged, comprehensive ExtractorState."""
    workflow = build_phase1_graph()
    app = workflow.compile()

    initial = ExtractorState(
        document=document_text,
        sentences=[],
        datapoints=DataPoints(skills=[]),
    )

    result = app.invoke(initial)
    final_state = ExtractorState.model_validate(result)

    return final_state

