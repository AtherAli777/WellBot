import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from textblob import TextBlob
import os
from streamlit_chat import message

# Initialize session state variables
def init_session_state():
    session_vars = {
        "messages": [],
        "chat_history": [],
        "vectorstore": None,
        "retriever": None,
        "api_key": None,
        "model": None,
        "crisis_score": 0,
        "client_name": None
    }
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
init_session_state()

# LazyLoadedFAISS class with caching
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

def sentiment_analysis(sentiment):
    text = TextBlob(sentiment)
    return text

def check_risk_keywords(text):
    risk_keywords = ['suicide', 'kill myself', 'end it all', 'no reason to live']
    return any(keyword in text.lower() for keyword in risk_keywords)

def update_crisis_score(text, sentiment):
    if check_risk_keywords(text):
        st.session_state.crisis_score += 2
    if sentiment.polarity < -0.5:
        st.session_state.crisis_score += 1
    return st.session_state.crisis_score

def provide_helpline_info():
    return """
    If you're feeling suicidal, please reach out for help:
    - National Suicide Prevention Lifeline: 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    Remember, you're not alone, and help is available 24/7.
    """

# Set page config
st.set_page_config(page_title="WellBot", layout="wide")

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

    # if st.button("New Chat", key="new_chat_button"):
    #     st.session_state.messages = []
    #     st.session_state.chat_history = []

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

def initialize_retriever():
    llm = get_llm(st.session_state.model, 0.3)
    
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = LazyLoadedFAISS("./faiss_index")

    template = """You are an AI assistant embodying the role of a professional therapist named Dr. Alina. As Dr. Alina, you have years of experience in counseling and are known for your empathy, patience, and ability to create a safe space for clients. Your primary goal is to understand, support the user and detect suicide ideation, not to immediately solve their problems.

    Adhere to these guidelines in your role as Dr. Alina:

    1. Conversation Flow:
    - If this is the first interaction (the client's name is not known), introduce yourself and ask for the client's name.
    - In subsequent interactions, use the client's name and refer to previous context when appropriate.

    2. Professional Demeanor:
    - Maintain a calm, compassionate, and non-judgmental tone at all times.
    - Use professional language, but avoid jargon that might confuse the client.

    3. Therapeutic Approach:
    - Practice active listening, reflecting the client's feelings and thoughts.
    - Ask open-ended questions to encourage the client to elaborate.
    - Validate the client's emotions and experiences.
    - Avoid giving direct advice; instead, guide the client to their own insights.

    4. Safety Considerations:
    - If the client expresses thoughts of self-harm or harming others, prioritize their safety and provide appropriate resources.

    Remember: Your role is to guide, support, and empower the client, not to solve their problems for them.

    Context: {context}
    Chat History: {chat_history}
    Human: {question}

    Dr. Alina: """

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

# Main chat functionality
if st.session_state.api_key:
    try:
        if st.session_state.retriever is None:
            initialize_retriever()
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
            with st.spinner("Generating response..."):
                try:
                    # Check if retriever is None and reinitialize if necessary
                    if st.session_state.retriever is None:
                        initialize_retriever()
                    
                    sentiment = sentiment_analysis(user_input)
                    crisis_score = update_crisis_score(user_input, sentiment)
                    if st.session_state.crisis_score > 5:
                        helpline_info = provide_helpline_info()
                        st.warning(helpline_info)
                        st.session_state.crisis_score = 0

                    retriever_input = f"Client's name: {st.session_state.client_name if st.session_state.client_name else 'Unknown'}. {user_input}"         
                    result = st.session_state.retriever({"question": retriever_input})

                    ai_response = result['answer']
                    # Check if the AI is asking for the name
                    if "may I ask your name?" in result['answer'].lower() and st.session_state.client_name is None:
                        st.session_state.client_name = user_input
                    
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.chat_history.append((user_input, ai_response))
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
        st.session_state.client_name = None
        st.session_state.vectorstore = None  
else:
    st.warning("Please enter your OpenAI API key in the sidebar to start the chat.")

# Footer
st.markdown("Made by Ather Ali")