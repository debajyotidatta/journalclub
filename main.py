from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Pinecone, VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
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
import os
import sys


def load_index():
    """Loads the index."""

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
        environment="us-east1-gcp"  # next to api key in console
    )
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return Pinecone.from_existing_index("hackathon", embeddings)


def query_index(llm, index, query):
    """Returns top 10 chunks."""
    num_matches = 10

    docs = [d[0]
            for d in index.similarity_search_with_score(query, k=num_matches)]

    # docs = [
    #     Document(page_content="My favorite fall vegetable is a sweet potato."),
    #     Document(page_content="Sweet potatoes are spherical."),
    #     Document(page_content="Fall is better than spring."),
    # ]
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



index = load_index()
chain = load_chain(index)
user_input = sys.argv[1]

print(user_input)

prompt = user_input

output = chain.run(input=prompt)
print(output)
