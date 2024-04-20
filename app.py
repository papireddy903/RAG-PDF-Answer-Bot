import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.prompts import PromptTemplate 
from langchain.chains import RetrievalQA

global retriever
global llm
global embeddings
global file_path

llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
file_path = "faiss_index"

def pdf_embed(pdf_path):
    # print(pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    print(pages[0])

    faiss_index = FAISS.from_documents(pages, embedding=embeddings)
    faiss_index.save_local(file_path)

    vectordb = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

    global retriever
    retriever = vectordb.as_retriever()

response_template = PromptTemplate(
    input_variables = ["context","question"],
    template = """ 
        Given the following context and question, generate answer based on this context only. 
        if you don't find answer in the given context. just make up response like as an AI language model something
        context : {context}
        question : {question}

    """
)

def qa_chain():
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt":response_template}
    )
    return chain 

# Declare variables as global

