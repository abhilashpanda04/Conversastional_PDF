import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_vector_store(text):
    pass
def get_text_chunks(text):
    """
    Splits a given string into chunks of 1000 characters, with a 200 character overlap
    between each chunk. Returns a list of these chunks.
    
    :param text: The string to be split into chunks.
    :type text: str
    :return: A list of text chunks.
    :rtype: List[str]
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks
def get_pdf_text(pdf_docs):
    """
    Gets the text from PDF documents.

    Args:
        pdf_docs (list): A list of file paths to PDF documents.

    Returns:
        str: The text extracted from all pages of all PDF documents.
    """
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def main():
    load_dotenv()
    
    st.set_page_config(page_title="chat-PDF",page_icon=":books:")
    st.header("")
    st.text_input("Ask a question based on the PDF")
    with st.sidebar:
        st.subheader("Your documents")
        pdf_files=st.file_uploader("Upload your files and click submit",accept_multiple_files=True)
        if st.button("submit"):
            with st.spinner("processing...."):
            
                #get pdf text

                raw_text=get_pdf_text(pdf_files)

                #get text chunks
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)

                #create a vector store
                vectors=get_vector_store(text_chunks)



if __name__=="__main__":
    main()
