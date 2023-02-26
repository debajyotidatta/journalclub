from typing import Any

from langchain.llms import OpenAI


def get_prediction_without_memory(text: str, model: Any = OpenAI(temperature=0)) -> str:
    """Get the prediction from the model."""
    return model(text)
