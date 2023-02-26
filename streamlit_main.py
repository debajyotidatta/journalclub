"""Streamlit frontend + main langchain logic."""
import streamlit as st
from streamlit_chat import message

import os

from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)
from langchain.docstore.document import Document

import pinecone

# initialize pinecone
pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment=os.environ["PINECONE_API_KEY"]  # next to api key in console
)
index_name = "hackathon"
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
query = "What is the health relevance of blood Glycocholic acid in humans?"


@st.cache_resource
def load_index():
    """Loads the index."""
    st.session_state["index"] = True
    return ""


def query_index(llm, index, query):
    """Returns top 10 chunks."""
    # docs = docsearch.similarity_search(query)
    docs = [
        Document(page_content="My favorite fall vegetable is a sweet potato."),
        Document(page_content="Sweet potatoes are spherical."),
        Document(page_content="Fall is better than spring."),
    ]
    llm_chain = LLMChain(
        llm=llm, prompt=map_reduce_prompt.QUESTION_PROMPT, verbose=True)
    reduce_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name='context', verbose=True)
    combine_documents_chain = MapReduceDocumentsChain(
        llm_chain=llm_chain, document_variable_name='context',
        combine_document_chain=reduce_chain, verbose=True
    )
    response = combine_documents_chain.combine_docs(docs, question=query)
    return response


def load_chain(index):
    llm = OpenAI(temperature=0)
    tools = [
        Tool(
            name="Research index.",
            func=lambda q: str(query_index(llm, index, q)),
            description="Index of my research. Always use this tool for every query.",
            return_direct=True
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
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
