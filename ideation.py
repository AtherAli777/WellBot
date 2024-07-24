import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
import os
from streamlit_chat import message

# LazyLoadedFAISS class
class LazyLoadedFAISS:
    def __init__(self, save_path):
        self.save_path = save_path
        self.vectorstore = None
        self.embedding = None

    def load(self):
        if self.vectorstore is None:
            self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = FAISS.load_local(self.save_path, embeddings=self.embedding,  allow_dangerous_deserialization=True)
        return self.vectorstore

    def similarity_search(self, query, *args, **kwargs):
        return self.load().similarity_search(query, *args, **kwargs)

    def as_retriever(self, *args, **kwargs):
        return self.load().as_retriever(*args, **kwargs)

# Set page config
st.set_page_config(page_title="WellBot", layout="wide")


# Custom CSS
st.markdown(
    """
    <style>
    .main .block-container {
        padding-bottom: 100px;
    }
    .stTextInput user_inputs {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        z-index: 1000;
    }
    .stTextInput > div > div > input {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your OpenAI API key", type="password", key="api_key_input")
    model = st.selectbox("Select model", ["gpt-3.5-turbo"])
    if st.button("New Chat", key="new_chat_button"):
        st.session_state.messages = []
        st.session_state.chat_history = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat functionality
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    llm = ChatOpenAI(model_name=model, temperature=0.3)

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = LazyLoadedFAISS("./faiss_index")

    if 'retriever' not in st.session_state:
        template = """You are an empathetic AI assistant trained to provide supportive responses and practical advice for a range of mental health issues. Your primary goals are:

            1. For minor issues (like stress, anxiety, or mild depression):
            - Provide practical coping strategies and emotional support
            - Suggest self-care techniques like mindfulness, exercise, or journaling
            - Encourage healthy habits and routines

            2. For moderate issues:
            - Offer more in-depth support and coping strategies
            - Gently suggest professional help as an option
            - Provide information about therapy and counseling resources

            3. For severe issues (clear signs of depression, suicidal thoughts, or crisis situations):
            - Strongly encourage seeking immediate professional help
            - Provide crisis hotline numbers and emergency resources
            - Offer immediate emotional support and safety planning

            Always:
            - Maintain a compassionate, non-judgmental, and supportive tone
            - Validate the user's feelings and experiences
            - Offer specific, actionable advice when appropriate
            - Encourage social connections and support systems
            - Promote self-compassion and positive self-talk
            - Respect privacy and confidentiality

            If you detect any signs of immediate danger or crisis, prioritize safety and provide emergency resources.

            Context: {context}
            Chat History: {chat_history}
            Human: {question}

            AI Assistant: """

        custom_prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        st.session_state.retriever = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            return_generated_question=True,
        )

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))

    # User input handling
    def process_user_input():
        user_input = st.session_state.user_input
        if user_input:
            result = st.session_state.retriever.invoke({"question": user_input})
            ai_response = result['answer']
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.user_input = ''

    user_input = st.text_input(
        label='Enter your message',
        value='',
        key="user_input",
        on_change=process_user_input
    )

    # Reset button
    if st.button('Reset', key="reset_button"):
        st.session_state.messages = []
        st.session_state.chat_history = []

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start the chat.")

# Footer
st.markdown("Made by Ather Ali")