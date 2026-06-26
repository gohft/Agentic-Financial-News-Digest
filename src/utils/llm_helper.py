from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


def get_llm(provider: str, model_args: dict[str, Any]) -> BaseChatModel:
    """
    Initialize and return an LLM based on the provider and model arguments.

    Args:
        provider: Name that matches the library that will load the LLM.
        model_args: Dictionary of model arguments, keys depends on model to use.

    Returns:
        A LangChain LLM object initialized with the specified model arguments.

    Raise:
        ValueError if the provider does match any of the supported providers.
    """
    # dictionary map between name of provider and the corresponding library.
    # add more providers if using models from different libraries like OpenAI
    providers = {
        "ollama": ChatOllama,
    }

    provider = provider.strip().lower()

    if provider not in providers:
        raise ValueError(
            f"Unsupported provider: {provider}, supported values are {', '.join(providers)}."
        )

    # return model with the model arguments as input
    return providers[provider](**model_args)
