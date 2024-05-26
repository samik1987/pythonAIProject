import requests
from bs4 import BeautifulSoup
import openai
import streamlit as st

# Set your OpenAI API key
openai.api_key = "*******"

# Function to scrape content from a website
def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text content from the webpage
    content = ' '.join([p.text for p in soup.find_all('p')])
    return content

# Function to generate response from OpenAI API
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Main function
def main():
    # URL of the website to scrape
    url = 'https://www.dakshineswarkalitemple.org/history.html'
    # Scrape content from the website
    content = scrape_content(url)
    # Query prompt to send to OpenAI API
    query_prompt = "Who built Dakshnieswar Kali Temple and when it was ?"
    # Generate response from OpenAI API
    response = generate_response(query_prompt)
    # Print the generated response
    print("Generated Response:")
    print(response)

button = st.button("Lets Query :")

if button:
    main()
