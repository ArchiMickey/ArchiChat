import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from ollama import Client
from langchain_community.llms.ollama import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from loguru import logger
from rich import print as rprint
from streamlit_extras.bottom_container import bottom
from langchain_core.messages import AIMessage, HumanMessage

def get_doc_intro_chain(llm):
    intro_prompt = """You are an assistant for paper reading tasks. \
    Use the following pieces of context to introduce this papaer. \
    Use three sentences maximum and keep the introduction concise.\
    Suggest 4 questions that users might ask about the paper in numbered list.\


    {context}"""
    intro_prompt = ChatPromptTemplate.from_template(intro_prompt)
    intro_chain = create_stuff_documents_chain(llm, intro_prompt)
    return intro_chain

def get_rag_qa_chain(llm, vectorstore):
    context_q_system_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, vectorstore.as_retriever(), context_q_prompt)
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

def get_qa_chain(llm):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of context to answer the question. \
    If you don't know the answer, just say that you don't know."""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = qa_prompt | llm
    return qa_chain

def split_document(document):
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splitted_doc = text_splitter.split_documents(document)
    return splitted_doc

def parse_history(messages):
    ret = ""
    for message in messages[-9:-1]:
        if message["role"] == "user":
            ret += f"Human: {message['content']}\n"
        else:
            ret += f"AI: {message['content']}\n"
    return ret

def response_parser(response):
    for r in response:
        if "answer" in r:
            yield r["answer"]    

def main():
    st.set_page_config(page_title="ArchiChat", page_icon="🤖", layout="wide")
    # Setup
    OLLAMA_HOST = st.sidebar.text_input("Ollama Host", value="localhost:11434")
    client = Client(host=OLLAMA_HOST)
    available_model = [modelfile["name"] for modelfile in client.list()["models"]]
    available_model.sort()
    
    selected_model = st.sidebar.selectbox("Select Model", available_model)
    llm = Ollama(model=selected_model)
    
    st.title("ArchiChat - A PDF Chatbot")
    st.write("Welcome to ArchiChat!")
    
    
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
    for message in st.session_state.history.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)    
    
    loader = None
    if st.sidebar.button("Load PDF"):
        if uploaded_file is not None:
            with st.spinner("Uploading PDF..."):
                temp_file = "./temp.pdf"
                with open(temp_file, "wb") as file:
                    file.write(uploaded_file.getvalue())
            logger.info(f"Uploaded file saved to {temp_file}")
            
            if not selected_model:
                st.error("Please select a model first!")
            else:
                loader = PyMuPDFLoader(temp_file)
                with st.spinner("Loading PDF..."):
                    document = loader.load()
                with st.spinner("Splitting document..."):
                    first_page = document[0]
                    splitted_doc = split_document(document)
                    logger.info(f"Splitted document into {len(splitted_doc)} chunks")
                with st.spinner("Creating embeddings..."):
                    embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": "cuda"})
                with st.spinner("Creating vectorstore..."):
                    vectorstore = Chroma.from_documents(splitted_doc, embedding_model)
                    st.session_state.vectorstore = vectorstore
                    
                intro_chain = get_doc_intro_chain(llm)
                response = intro_chain.stream({"context": [first_page]})
                with st.chat_message("ai"):
                    response = st.write_stream(response)
                st.session_state.history.add_message(AIMessage(response))
            
        else:
            st.error("Please upload a PDF file first!")

    
    qa_chain = get_qa_chain(llm)
    conversational_chain = RunnableWithMessageHistory(
        qa_chain,
        lambda x: st.session_state.history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    if "vectorstore" in st.session_state:
        rag_chain = get_rag_qa_chain(llm, st.session_state.vectorstore) 
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda x: st.session_state.history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.history.clear()
        st.rerun()

    
    user_input = st.chat_input("Say something to the chatbot")      
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if "vectorstore" in st.session_state:
            response = conversational_rag_chain.stream(
                {"input": user_input},
                config={
                    "configurable": {"session_id": 0}
                }
            )
        else:        
            response = conversational_chain.stream(
                {"input": user_input},
                config={
                    "configurable": {"session_id": 0}
                }
            )
        
        with st.chat_message("ai"):
            response = st.write_stream(response_parser(response))

if __name__ == "__main__":
    main()