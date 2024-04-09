import streamlit as st
from streamlit_chat import message
from researcher import InfoResearcher
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=True)
def instantiate_researcher():
    # Instantiate the InfoResearcher
    info_researcher = InfoResearcher()
    return info_researcher

research_helper = instantiate_researcher()

def show_discussion(history):
    # Display the conversation history
    for index in range(len(history["helper"])):
        message(history["user"][index], is_user=True, key=str(index) + "_user")
        message(history["helper"][index], key=str(index))

# Initialize `clicked` in session_state if it doesn't exist
if "clicked" not in st.session_state:
    st.session_state.clicked = True  # or False, depending on your needs

if st.session_state.clicked:
    st.title("AI Research Assistant 🧑‍💻")
    st.subheader("Instant, precise answers for any query, powered by AI.")

    if "helper" not in st.session_state:
        st.session_state["helper"] = ["Greetings! How may I assist you today?"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hello!"]

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.expander("Interact with the AI"):
            query_input = st.text_input("Enter your query")
            if st.button("Submit"):
                ai_response = research_helper.conduct_research(query_input)

                st.session_state["user"].append(query_input)
                st.session_state["helper"].append(ai_response)

                if st.session_state["helper"]:
                    show_discussion(st.session_state)
