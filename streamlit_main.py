"""Streamlit frontend + main langchain logic."""
import streamlit as st
from streamlit_chat import message

from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI


llm = OpenAI()


@st.cache_resource
def load_index():
    """Loads the index."""
    st.session_state["index"] = True
    return ""


def query_index(index, query):
    return "My favorite fall vegetable is a sweet potato."


def load_chain(index):
    llm = OpenAI(temperature=0)

    tools = [
        Tool(
            name="Index of my research ",
            func=lambda q: str(query_index(index, q)),
            description="Index of my research",
            return_direct=True
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory, verbose=True)
    return agent_chain


def get_text():
    input_text = st.text_input(
        "Enter your question here: ", "", key="input")
    return input_text


st.set_page_config(
    page_title="Journal Club", page_icon=":robot:")
st.header("Journal Club")
st.subheader("Journal Club")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


index = load_index()

if "index" in st.session_state:
    chain = load_chain(index)
    user_input = get_text()

    if user_input:
        prompt = "Call GPT Index: " + user_input
        print("\nFull Prompt:\n", prompt)
        output = chain.run(input=prompt)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")
