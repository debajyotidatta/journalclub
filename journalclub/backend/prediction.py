from typing import Any

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI


def get_prediction_without_memory(text: str, model: Any = OpenAI(temperature=0)) -> str:
    """Get the prediction from the model."""
    return model(text)


# def load_chain(index, llm=OpenAI(temperature=0)) -> Any:
#     tools = [
#         Tool(
#             name="Research index.",
#             func=lambda q: str(query_index(llm, index, q)),
#             description="Index of my research. Always use this tool for every query.",
#             return_direct=True,
#         ),
#     ]
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     agent_chain = initialize_agent(
#         tools,
#         llm,
#         agent="conversational-react-description",
#         memory=memory,
#         verbose=True,
#     )
#     return agent_chain


def load_conversation_memory():
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )
    return conversation
