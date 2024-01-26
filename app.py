from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv,find_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle 
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

## Slide-bar
with st.sidebar:
    st.title('PDF Q&A')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by Harshit')
    
def main():
    st.header("Q&A from Pdfs: ")
    
    
    load_dotenv(find_dotenv())
    
    pdf_reader = PdfReader('48lawsofpower.pdf')
    # st.write(pdf_reader)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )
    ## Chunk Formation
    chunks = text_splitter.split_text(text= text)
    
    ## Embedding    
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(chunks, embeddings)
    
    
    query = st.text_input("Ask your questions: ")
     
    docs = document_search.similarity_search(query=query)
 
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.write(response)
    
        
if __name__ == '__main__':
    main()
    
    