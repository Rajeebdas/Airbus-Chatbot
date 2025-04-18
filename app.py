
import streamlit as st
import numpy as np
import pickle
import json
import random
import datetime
import requests
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and preprocessing assets
model = load_model('chatbot_model.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Get max sequence length used during training
max_length = model.input_shape[1]

# News API
def get_latest_news():
    news_api_key = 'fb0bfd4c2cf14463b04f68cff6390a22'
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={news_api_key}"
    response = requests.get(url).json()
    headlines = [article['title'] for article in response.get('articles', [])[:5]]
    return "\n".join(headlines) if headlines else "Sorry, I couldn't fetch the news."

# Get chatbot response
def chatbot_response(text):
    words = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]
    seq = tokenizer.texts_to_sequences([" ".join(words)])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')

    prediction = model.predict(padded_seq, verbose=0)[0]
    tag = classes[np.argmax(prediction)]

    for intent in data['intents']:
        if intent['tag'] == tag:
            if tag == "date":
                return str(datetime.date.today())
            elif tag == "time":
                return datetime.datetime.now().strftime("%H:%M:%S")
            elif tag == "latest_news":
                return get_latest_news()
            else:
                return random.choice(intent['responses'])

# Streamlit App Setup
st.set_page_config(page_title="Chatbot 🤖", page_icon="💬")
st.title("🤖 LSTM Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chatbot_response(prompt)

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
