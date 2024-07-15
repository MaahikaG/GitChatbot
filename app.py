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

# Access the API keys
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")

def create_chain (vectorStore):
    #Instantiate LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        # repo_id="openai-community/gpt2",
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
    namespace = "command_list"

    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embedding,
        namespace=namespace
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
        

