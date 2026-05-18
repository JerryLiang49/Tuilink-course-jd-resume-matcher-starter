"""OpenAI chat and embedding client helpers with optional local caching."""

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
# Load local .env values for notebooks/local runs. In Lambda, these values should
# be provided through environment variables by the infra stack.
load_dotenv()

# Get API key from environment variable
# The OpenAI key is required at import time because the default chat and
# embedding clients are constructed below.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


# Cache is useful during notebooks because repeated prompts can be expensive.
# For Lambda deployment this should usually be false because local filesystem
# cache files are ephemeral and can cause confusing behavior across warm starts.
LLM_USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

# Model selection and temperature
# Chat model settings. Temperature may be ignored for models that do not support
# it, handled by _model_supports_temperature below.
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
try:
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0").strip())
except ValueError:
    LLM_TEMPERATURE = 0.0


# Embedding model is separate from the chat model because matching phases may use
# vector similarity even when extraction uses a chat model.
LLM_EMBEDDING_MODEL: str = os.getenv(
    "LLM_EMBEDDING_MODEL", "text-embedding-3-small"
).strip()

logging.info(f"LLM_MODEL: {LLM_MODEL}")
logging.info(f"LLM_TEMPERATURE: {LLM_TEMPERATURE}")
logging.info(f"LLM_USE_CACHE: {LLM_USE_CACHE}")
logging.info(f"LLM_EMBEDDING_MODEL: {LLM_EMBEDDING_MODEL}")


def _model_supports_temperature(model_name: str) -> bool:
    """Return whether this model should receive a temperature argument."""

    # Newer reasoning-style models like gpt-5 do not accept temperature
    return not (model_name.startswith("gpt-5"))


# Only pass temperature when the model accepts it. Passing unsupported parameters
# can make OpenAI requests fail before any workflow logic runs.
temperature_arg: Optional[float] = (
    LLM_TEMPERATURE if _model_supports_temperature(LLM_MODEL) else None
)

# Default clients are shared by all calls. This avoids rebuilding clients for
# every node invocation in notebooks or warm Lambda containers.
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
    """Invoke the chat LLM with optional file-backed caching.

    Cache keys include the message input and runnable config. They intentionally
    do not include arbitrary kwargs, so use cache only when kwargs are stable or
    irrelevant for the prompt result.
    """

    if use_cache:
        # Create a hash of the input string
        # Create a stable local file path for this prompt/config combination.
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
        # Direct model call path used by deployment and one-off uncached runs.
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
        # Include the embedding model in the cache key so changing models cannot
        # accidentally reuse vectors from an older dimensionality/space.
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
            # Fetch from OpenAI and store as float32 to reduce local cache size.
            embedding_list = embeddings.embed_query(text, **kwargs)
            embedding_array = np.array(embedding_list, dtype=np.float32)

            write_file(filepath, embedding_array, verbose=verbose)
            return embedding_array
    else:
        # Direct embedding path for callers that want fresh vectors every time.
        embedding_list = embeddings.embed_query(text, **kwargs)
        return np.array(embedding_list, dtype=np.float32)
