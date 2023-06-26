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
    """
    Initializes a conversational retrieval chain from a given vectorstore. The chain first uses a ChatOpenAI model to generate
    responses and then retrieves appropriate responses from a vectorstore. The memory parameter specifies the conversation
    buffer memory to use for storing chat history. Returns the initialized conversational retrieval chain.
    
    :param vectorstore: A vectorstore containing the knowledge base for the retrieval chain.
    :type vectorstore: VectorStore
    :return: A conversational retrieval chain initialized with the given vectorstore and memory.
    :rtype: ConversationalRetrievalChain
    """
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversastion_chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),memory=memory)
    return conversastion_chain
def get_vector_store(text_chunks):
    """
    Returns a vector store for the given text chunks.
    
    Args:
        text_chunks (list): A list of text chunks to be used for creating the vector store.
        
    Returns:
        vector_store (FAISS): A FAISS object representing the vector store generated from the given text chunks.
    """
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
    """
    Handles a user question by passing it to the chatbot and displaying the response. 

    Args:
    - user_question (str): The question asked by the user. 

    Returns:
    - the response from the chatbot.
    """
    response=st.session_state.conversastion({'question':user_question})
    st.session_state.chat_history=response['chat_history']
    for i , message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
def main():
    """
    This function is the main entry point for the chat-PDF application.
    It loads the environment variables, sets the page configuration, and 
    initializes the conversation and chat history in the session state.
    Then it prompts the user for a question based on the PDF
    and handles the user's question by calling the 'handle_user_question' function.
    The function also creates a file uploader in the sidebar for the user to upload their PDF files 
    and a 'submit' button to start the processing. Once the user submits their PDF files, the function gets the raw text from the PDF files,
    creates text chunks, and generates a vector store.
    Finally, the function stores the conversation chain in the session state and returns nothing.
    """

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
