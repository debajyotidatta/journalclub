# from langchain.agents import Tool, initialize_agent
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain.llms import OpenAI

# llm = OpenAI()


# def load_chain(index):

#     tools = [
#         Tool(
#             name="Index of my research ",
#             # func=lambda q: str(query_index(index, q)),
#             description="Index of my research",
#             return_direct=True,
#         ),
#     ]
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     llm = OpenAI(temperature=0)
#     agent_chain = initialize_agent(
#         tools,
#         llm,
#         agent="conversational-react-description",
#         memory=memory,
#         verbose=True,
#     )
#     return agent_chain


# output = chain.run(input=prompt)
