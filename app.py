import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
import hashlib
import urllib.parse
import firebase_admin
from firebase_admin import credentials, auth
import json

# Access the API keys
# Debug: Show what secrets are available
st.write("Debug Info:")
st.write("Available secrets:", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets object")
st.write("Environment variables with HF:", [k for k in os.environ.keys() if 'HF' in k.upper()])
st.write("Environment variables with TOKEN:", [k for k in os.environ.keys() if 'TOKEN' in k.upper()])

# Try multiple ways to get the token
try:
  huggingfacehub_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
  st.write("✅ Found token in st.secrets")
except Exception as e:
  st.write("❌ Error accessing st.secrets:", str(e))
  try:
      huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
      st.write("✅ Found token in environment variables")
  except Exception as e2:
      st.write("❌ Error accessing environment:", str(e2))
      st.stop()

pinecone_api_key = st.secrets["PINECONE_API_KEY"]

os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Firebase Admin
@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        # Use the same Firebase project as your client
        firebase_config = {
            "type": "service_account",
            "project_id": "version-wise",
            "private_key_id": st.secrets["FIREBASE_PRIVATE_KEY_ID"],
            "private_key": st.secrets["FIREBASE_PRIVATE_KEY"].replace('\\n', '\n'),
            "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
            "client_id": st.secrets["FIREBASE_CLIENT_ID"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["FIREBASE_CLIENT_CERT_URL"]
        }
        
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    return firebase_admin

# Initialize Firebase
initialize_firebase()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Get student information from URL parameters and Firebase token
query_params = st.experimental_get_query_params()
firebase_token = query_params.get("token", [None])[0]

student_id = None
student_email = None
student_namespace = "git_book"  # Default namespace

if firebase_token:
    try:
        # Verify the Firebase token
        decoded_token = auth.verify_id_token(firebase_token)
        student_id = decoded_token.get('uid')
        student_email = decoded_token.get('email', 'unknown@email.com')
        
        # Create student-specific namespace
        student_identifier = hashlib.md5(f"{student_id}".encode()).hexdigest()[:8]
        student_namespace = f"student_{student_identifier}"
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        student_id = None
        student_email = None

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")

def create_chain (vectorStore):
    #Instantiate LLM
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # chain = prompt | llm
    output_parser = StrOutputParser()
    chain = create_stuff_documents_chain (
        llm = llm,
        prompt = prompt,
        output_parser = output_parser
    )

    retriever = vectorStore.as_retriever()

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )


    retrieval_chain = create_retrieval_chain (
        history_aware_retriever, 
        chain
    )
    return retrieval_chain

def process_chat (chain, question, chat_history):
    response = chain.invoke ({
        "chat_history": chat_history,
        "input": question
    })
    return response["answer"]

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)



if __name__ == '__main__':
    # Connect to the index
    index_name = "versionwise"
    index = pc.Index(index_name)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embedding,
        namespace="git_book"
    )

    chain = create_chain(vectorstore)

    chat_history = []

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            response = process_chat(chain, user_query, st.session_state.chat_history)
            st.write(response)  
        st.session_state.chat_history.append(AIMessage(content=response))
        

