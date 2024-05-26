from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import requests
import re

OPENAI_API_KEY = "*******" #Pass your key here
#Upload PDF files
st.header("Samik's AI Chatbot")
with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")
def extract_text_from_html(web_page):
    # REQUEST WEBPAGE AND STORE IT AS A VARIABLE
    page_to_scrape = requests.get(web_page)
    # USE BEAUTIFULSOUP TO PARSE THE HTML AND STORE IT AS A VARIABLE
    soup = BeautifulSoup(page_to_scrape.text, 'html.parser')
    text = soup.get_text()
    return text

def remove_extra_spaces(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Create an input textbox for text input
input_textbox = st.text_input("Enter Website :")
user_question = st.text_input("Type Your question here :")
# Create a button to trigger input retrieval
buttonGrabContent = st.button("Lets grab")
cleaned_text = ""
#Extract the text
if file is not None or buttonGrabContent:

    if file is not None and input_textbox == "":
        input_textbox = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            cleaned_text += page.extract_text()
            #st.write(text)
    else:
        file =  None
        text_content = extract_text_from_html(input_textbox)
        cleaned_text = remove_extra_spaces(text_content)

    if cleaned_text != "":
        #Break it into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        #st.write("Input:", cleaned_text)
        chunks = text_splitter.split_text(cleaned_text)
        #st.write(chunks)

        #st.write("1")
        # generating embedding
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # creating vector store - FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)

        # get user question
        #user_question = st.text_input("Type Your question here :")

        #st.write("2")
        # do similarity search
        if user_question:
            #st.write("3")
            match = vector_store.similarity_search(user_question)
            #st.write("match :",match)
            #st.write("4")
            #define the LLM
            llm = ChatOpenAI(
                openai_api_key = OPENAI_API_KEY,
                temperature = 0,
                max_tokens = 1000,
                model_name = "gpt-3.5-turbo"
            )
            # Searched Text chunk from Vector dB by user query
            # st.write(match)
            #output results
            #chain -> take the question, get relevant document, pass it to the LLM, generate the output
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = match, question = user_question)
            st.write(response)