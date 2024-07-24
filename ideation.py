import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
import os
from streamlit_chat import message


# Initialize session state variables
def init_session_state():
    session_vars = [
        "messages",
        "chat_history",
        "vectorstore",
        "retriever",
        "api_key",
        "model"
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
init_session_state()

class LazyLoadedFAISS:
    def __init__(self, save_path):
        self.save_path = save_path
        self.vectorstore = None
        self.embedding = None

    @staticmethod
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @staticmethod
    @st.cache_resource
    def load_vectorstore(save_path, _embeddings):
        return FAISS.load_local(save_path, embeddings=_embeddings, allow_dangerous_deserialization=True)

    def load(self):
        if self.vectorstore is None:
            try:
                self.embedding = self.load_embeddings()
                self.vectorstore = self.load_vectorstore(self.save_path, self.embedding)
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
                return None
        return self.vectorstore

    def similarity_search(self, query, *args, **kwargs):
        try:
            return self.load().similarity_search(query, *args, **kwargs)
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []

    def as_retriever(self, *args, **kwargs):
        try:
            return self.load().as_retriever(*args, **kwargs)
        except Exception as e:
            st.error(f"Error creating retriever: {str(e)}")
            return None

# Set page config
st.set_page_config(page_title="WellBot", layout="wide")

# ... [Your existing custom CSS code remains unchanged] ...

# Sidebar
with st.sidebar:
    st.title("Settings")
    new_api_key = st.text_input("Enter your OpenAI API key", type="password", key="api_key_input")
    new_model = st.selectbox("Select model", ["gpt-3.5-turbo"])

    # Update session state only if values have changed
    if new_api_key != st.session_state.api_key:
        st.session_state.api_key = new_api_key
        os.environ['OPENAI_API_KEY'] = new_api_key
        # Reset chat-related session state when API key changes
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.retriever = None

    if new_model != st.session_state.model:
        st.session_state.model = new_model
        # Reset retriever when model changes
        st.session_state.retriever = None

    if st.button("New Chat", key="new_chat_button"):
        st.session_state.messages = []
        st.session_state.chat_history = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cached function for creating ChatOpenAI instance
@st.cache_resource
def get_llm(_model_name, temperature):
    return ChatOpenAI(model_name=_model_name, temperature=temperature)

# Cached function for creating ConversationalRetrievalChain
@st.cache_resource
def get_conversation_chain(_llm, _retriever, _memory, _custom_prompt):
    return ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_retriever,
        memory=_memory,
        combine_docs_chain_kwargs={"prompt": _custom_prompt},
        return_source_documents=True,
        return_generated_question=True,
    )


# Main chat functionality
if st.session_state.api_key:
    try:
        if st.session_state.retriever is None:
            llm = get_llm(st.session_state.model, 0.3)
            
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = LazyLoadedFAISS("./faiss_index")

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
            vectorstore = st.session_state.vectorstore.load()
            if vectorstore is not None:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.session_state.retriever = get_conversation_chain(
                    llm,
                    retriever,
                    memory,
                    custom_prompt
                )
            else:
                st.error("Failed to load vector store")
                st.stop()

    except Exception as e:
        st.error(f"Error initializing chat components: {str(e)}")
        st.stop()

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))

    # User input handling
    def process_user_input():
        user_input = st.session_state.user_input
        if user_input:
            try:
                result = st.session_state.retriever.invoke({"question": user_input})
                ai_response = result['answer']
                
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.session_state.user_input = ''
            except Exception as e:
                st.error(f"Error processing user input: {str(e)}")

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
        st.session_state.retriever = None

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start the chat.")

# Footer
st.markdown("Made by Ather Ali")