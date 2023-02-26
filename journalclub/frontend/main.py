import streamlit as st
from streamlit_chat import message

# from journalclub.backend.prediction import get_prediction_without_memory,
# load_chain, load_conversation_memory
from journalclub.backend.prediction import load_chain


def initialize_state():
    if "responses" not in st.session_state:
        st.session_state["responses"] = []
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    # if chain is not set as a python variable
    if "chain" not in st.session_state:
        st.session_state["chain"] = load_chain()


def log_results(question, response):
    st.session_state.questions.append(question)
    st.session_state.responses.append(response)
    return response


def get_prediction(question):
    response = "Call GPT Index: " + question
    return response


def get_input():
    question = st.text_input("Enter your question here: ", "", key="input")
    return question


def display_response():
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"]) - 1, -1, -1):
            message(
                st.session_state["questions"][i], is_user=True, key="question_" + str(i)
            )
            message(st.session_state["responses"][i], key="response_" + str(i))


# create a chattbot like interface where inputs and responses get stored in the state
def main():
    st.set_page_config(page_title="Journal Club", page_icon=":robot:")
    st.header("Journal Club")

    initialize_state()

    question = get_input()

    if question:
        # response = get_prediction_without_memory(text=question)
        response = st.session_state["chain"].run(input=question)
        log_results(question, response)

    display_response()


if __name__ == "__main__":
    main()
