import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_components import css,bot_template,user_template
from langchain.chat_models import ChatOpenAI

def get_conversastion_chain(vectorstore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversastion_chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),memory=memory)
    return conversastion_chain
def get_vector_store(text_chunks):
    # embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings=OpenAIEmbeddings(model= "text-embedding-ada-002")
    vector_store=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vector_store
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


def handle_user_question(user_question):
    response=st.session_state.conversastion({'question':user_question})
    st.session_state.chat_history=response['chat_history']
    for i , message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
def main():

    load_dotenv()
    
    st.set_page_config(page_title="chat-PDF",page_icon=":books:")
    
    st.write(css,unsafe_allow_html=True)
    if "conversastion" not in st.session_state:
        st.session_state.conversastion=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    
    st.header("Chat with PDFs")
    user_question=st.text_input("Ask a question based on the PDF")
    if user_question:
        handle_user_question(user_question)

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
                vectorstore=get_vector_store(text_chunks)

                st.session_state.conversastion=get_conversastion_chain(vectorstore)




if __name__=="__main__":
    main()
