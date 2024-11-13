import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

##Load the Nvidia API Key

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')
print(os.environ['NVIDIA_API_KEY'])

st.title("Nvidia NIM Demo")
llm=ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct") ##NVIDIA NIM Inferencing

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./Documents") #Data Ingestion
        st.session_state.docs=st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) #Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #Splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vetor OpenAI embeddings

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided ontext only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    questions:{input}

    """
    )

prompt1=st.text_input("Enter your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("FAISS Vector store DB Is Ready Using NvidiaEmbedding")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input' :prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    #With a streamlit expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

