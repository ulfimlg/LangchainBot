from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile
import streamlit as st
import os


def process_pdf_and_question(pdf_file, question_text):
    if pdf_file.name.endswith('pdf'):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.getvalue())
            path = tmp.name
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    query = question_text
    result = db.similarity_search(query)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if the user finds the answer helpful.
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question_text})
    return response['answer']


def main():
    st.title("PDF Question Answering")

    if st.button("Reset"):
        st.session_state.clear()
        st.session_state.pdf_file = None
        st.session_state.question_text = ""

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    question_text = st.text_input("Enter your question")

    if pdf_file and question_text:
        answer = process_pdf_and_question(pdf_file, question_text)
        st.write(answer)


if __name__ == "__main__":
    main()
