import logging
import hashlib
import os
from typing import Any, Optional
import numpy as np
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.base import LanguageModelInput
from rexpand_pyutils_file import read_file, write_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


LLM_USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

# Model selection and temperature
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
try:
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0").strip())
except ValueError:
    LLM_TEMPERATURE = 0.0


LLM_EMBEDDING_MODEL: str = os.getenv(
    "LLM_EMBEDDING_MODEL", "text-embedding-3-small"
).strip()

logging.info(f"LLM_MODEL: {LLM_MODEL}")
logging.info(f"LLM_TEMPERATURE: {LLM_TEMPERATURE}")
logging.info(f"LLM_USE_CACHE: {LLM_USE_CACHE}")
logging.info(f"LLM_EMBEDDING_MODEL: {LLM_EMBEDDING_MODEL}")


def _model_supports_temperature(model_name: str) -> bool:
    # Newer reasoning-style models like gpt-5 do not accept temperature
    return not (model_name.startswith("gpt-5"))


temperature_arg: Optional[float] = (
    LLM_TEMPERATURE if _model_supports_temperature(LLM_MODEL) else None
)

default_llm = ChatOpenAI(
    model=LLM_MODEL, temperature=temperature_arg, api_key=OPENAI_API_KEY
)

default_embeddings = OpenAIEmbeddings(model=LLM_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def invoke_llm(
    input: LanguageModelInput,
    config: Optional[RunnableConfig] = None,
    *,
    use_cache: bool = LLM_USE_CACHE,
    verbose: bool = False,
    llm: Optional[ChatOpenAI] = default_llm,
    **kwargs: Any,
) -> BaseMessage:
    if use_cache:
        # Create a hash of the input string
        input_hash = hashlib.md5((str(input) + "|" + str(config)).encode()).hexdigest()
        filepath = f"./.cache/chats/{input_hash}.json"

        cached_response = read_file(filepath)
        if cached_response is not None:
            if verbose:
                logging.info(f"Cache hit: {filepath}")

            return AIMessage(**cached_response)
        else:
            if verbose:
                logging.info(f"Cache miss: {filepath}")

            response: BaseMessage = llm.invoke(input, config, **kwargs)
            write_file(filepath, response.model_dump())
            return response
    else:
        return llm.invoke(input, config, **kwargs)


def get_embedding(
    text: str,
    *,
    use_cache: bool = True,
    verbose: bool = False,
    embeddings: Optional[OpenAIEmbeddings] = default_embeddings,
    **kwargs: Any,
) -> np.ndarray:
    """
    Get embedding for a given text using OpenAI's embedding API.

    Args:
        text: The text to embed
        use_cache: Whether to use caching for embeddings
        verbose: Whether to log cache hits/misses
        embeddings: The OpenAI embeddings instance to use
        **kwargs: Additional arguments passed to the embeddings.embed_query method

    Returns:
        numpy array containing the embedding vector
    """
    if use_cache:
        # Create a hash of the input text and model
        input_hash = hashlib.md5((text + "|" + embeddings.model).encode()).hexdigest()
        filepath = f"./.cache/embeddings/{input_hash}.npy"

        cached_response = read_file(filepath, verbose=verbose)
        if cached_response is not None:
            if verbose:
                logging.info(f"Embedding cache hit: {filepath}")
            return cached_response
        else:
            if verbose:
                logging.info(f"Embedding cache miss: {filepath}")

            # Get embedding from OpenAI
            embedding_list = embeddings.embed_query(text, **kwargs)
            embedding_array = np.array(embedding_list, dtype=np.float32)

            write_file(filepath, embedding_array, verbose=verbose)
            return embedding_array
    else:
        embedding_list = embeddings.embed_query(text, **kwargs)
        return np.array(embedding_list, dtype=np.float32)
