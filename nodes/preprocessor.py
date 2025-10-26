import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.sentence import RawSentences, Sentence
from models.extractor_state import ExtractorState
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
)


def preprocessor(state: ExtractorState) -> ExtractorState:
    """Pre-process the input text into sentences."""
    if not state.document:
        raise ValueError("Document is empty")

    system_prompt = """
    You are a helpful assistant that splits the given text into sentences.
    Ignore meaningless sentences like title, header, footer,etc.
    """

    user_prompt = state.document

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

    state.sentences = [
        Sentence(id=index, sentence=sentence)
        for index, sentence in enumerate(
            json.loads(response.content[0]["text"])["sentences"]
        )
    ]
    return state
