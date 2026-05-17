"""LLM-backed preprocessing node that splits documents into sentences."""

import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.sentence import RawSentences, Sentence
from models.extractor_state import ExtractorState
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
)


def _response_json(response) -> dict:
    """Parse JSON from the LLM response across supported content formats."""

    content = response.content
    if isinstance(content, str):
        raw_text = content
    elif isinstance(content, list):
        raw_text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        )
    else:
        raw_text = str(content)

    return json.loads(raw_text)


def preprocessor(state: ExtractorState) -> ExtractorState:
    """Pre-process the input text into sentences."""

    # This is the first extraction step for both resumes and job descriptions.
    # The downstream skill extractor should operate on the numbered sentences
    # rather than the raw document so every skill can cite evidence.
    if not state.document:
        raise ValueError("Document is empty")

    state.phase = "phase1"
    state.step = "phase1:preprocessor"

    # Keep the prompt narrow: the LLM is not extracting skills here, only
    # deciding which meaningful sentence-like chunks should be retained.
    system_prompt = """
    You are a helpful assistant that splits the given text into sentences.
    Ignore meaningless sentences like title, header, footer,etc.
    """

    user_prompt = state.document

    # Force structured JSON output so parsing is deterministic. ``RawSentences``
    # contains only a list of strings; ids are assigned locally below.
    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={
            "format": {
                "name": "sentences",
                "strict": True,
                "type": "json_schema",
                "schema": RawSentences.model_json_schema(),
            }
        },
        use_cache=LLM_USE_CACHE,
    )

    # Convert the raw strings into stable Sentence objects. The generated index
    # becomes the evidence id used by later skill datapoints.
    state.sentences = [
        Sentence(id=index, sentence=sentence)
        for index, sentence in enumerate(_response_json(response)["sentences"])
    ]
    return state
