import streamlit as st
from bs4 import BeautifulSoup
import requests
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_input():
    # Retrieve the text from the input textbox
    user_input = input_textbox
    # Display the input
    if user_input:
        st.write("Input:", user_input)


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

OPENAI_API_KEY = "*********" #Pass your key here

# Streamlit UI
st.title("Input Text")

# Create an input textbox for text input
input_textbox = st.text_input("Enter text:")

# Create a button to trigger input retrieval
button = st.button("Get Input")

# When the button is clicked, call the get_input function
if button:
    get_input()
    text_content = extract_text_from_html(input_textbox)
    cleaned_text = remove_extra_spaces(text_content)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_text)
    #st.write("Input:", cleaned_text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type Your question here")

    #button1 = st.button("Lets Query :")
    if user_question: # Shift tab
        st.write("3")
        match = vector_store.similarity_search(user_question)
        # st.write("match :",match)
        st.write("4")
        # define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # output results
        # chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)

