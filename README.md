
# WellBot
Wellbot is a Retrieval-Augmented Generation (RAG) based chatbot project that uses Langchain and Streamlit to create an interactive AI assistant. The chatbot, embodying the role of Dr. Alina, a professional therapist, provides empathetic responses and support to users.

## Demo
You can access and interact with the live version of this project here: 
https://wellbot.streamlit.app/

## Features
- RAG-based responses using FAISS vector store
- Sentiment analysis for user inputs
- Crisis detection and helpline information
- Streamlit-based user interface
- Dynamic conversation flow with personalized responses

## Dataset
This project uses a custom-created dataset, `health.csv`, which forms the knowledge base for our RAG-based chatbot.

Dataset Description
- **Filename**: health.csv
- **Purpose**: To provide a diverse range of mental health scenarios and appropriate counselor responses.
- **Structure**: The dataset includes the following columns:
  - Scenario_ID: Unique identifier for each scenario
  - User_Statement: Simulated user input describing a mental health concern
  - Counselor_Response: Appropriate response from a mental health professional
  - Mental_Health_Issue: The type of mental health issue addressed in the scenario
  - Explanation: Brief explanation of the scenario and response

Dataset Creation
This dataset was carefully curated to cover a wide range of mental health topics, ensuring that our chatbot can provide relevant and supportive responses across various scenarios. The responses are designed to mimic those of a professional therapist, focusing on empathy, support, and appropriate guidance.

Usage in the Project
The `health.csv` file is used to create a FAISS index, which allows for efficient similarity search during the chatbot's operation. This enables the system to retrieve relevant information and generate contextually appropriate responses.

## Installation
To set up this project, follow these steps:
1. Clone the repository: https://github.com/AtherAli777/WellBot.git
2. Install the required dependencies: pip install -r requirements.txt

## Usage
To run the chatbot:
1. Set your OpenAI API key in the Streamlit sidebar.
2. Run the Streamlit app: streamlit run ideation.py
