import streamlit as st 
from app import pdf_embed, qa_chain
import os 

st.title("PDF Q/A Bot") 

question = st.text_input("Ask a question") 


pdf = st.sidebar.file_uploader("Upload PDF")

btn = st.button("Ask AI")

if pdf is not None:
    d_path = os.path.join(os.getcwd(), pdf.name)
    with open(d_path, 'wb') as f:
        f.write(pdf.getbuffer())
    pdf_embed(d_path) 

if question and pdf: 
    if btn:
        response = qa_chain().invoke(question)
        st.write(response['result'])
