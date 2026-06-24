from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(provider: str, model_args: dict[str, Any]) -> BaseChatModel:
    """
    Initialize and return an LLM based on the provider and model arguments.

    Args:
        provider:   "ollama" (case-insensitive)
        model_args: dictionary of model arguments.
    Returns:
        A LangChain chat model instance ready for use in a LangGraph node.
    """
    # dictionary map between name of provider and the langchain model
    providers = {
        # "openai": ChatOpenAI,
        "ollama": ChatOllama,
    }

    provider = provider.strip().lower()

    if provider not in providers:
        raise ValueError(
            f"Unsupported provider: {provider}, supported values are {', '.join(providers)}."
        )

    # return model with the model arguments as input
    return providers[provider](**model_args)
