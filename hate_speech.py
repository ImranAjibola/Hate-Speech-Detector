import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
# import urllib.request
# import csv
import os
api_key = os.environ.get('HF_TOKEN')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(r"C:\Users\Imran Ajibola\Desktop\RAIN\2nd semester\project\Hate Speech Detection\Hatespeech.csv", encoding= 'unicode_escape')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweets(tweet):
    # Remove links, RT, and @mentions
    tweet = re.sub(r'http\S+|RT|@\S+', '', tweet)
    # Remove non-alphanumeric characters except whitespace
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    # tokenize the text
    words = nltk.word_tokenize(tweet)
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    # lemmatize the text
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    # join the tokens back into a string
    preprocessed_text = ' '.join(words)
    return preprocessed_text

df['tweet'] = df['tweet'].apply(preprocess_tweets)

# Preprocess the tweet data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Train SVM classifier
svmc = SVC(random_state=1)
svmc.fit(X_train, y_train)
svm_preds = svmc.predict(X_test)
# function to classify input text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task = 'sentiment'

# Set the path to the folder containing the saved model and tokenizer
model_path = f'cardiffnlp/twitter-roberta-base-{task}'

# Set the name of the saved model file
model_name = 'pytorch_model.bin'

# Set the name of the saved tokenizer files
tokenizer_name = 'vocab'

# Load the saved tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=api_key)

# Load the saved model
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Function to classify input text
def classify_text(text):
    # preprocess the input text
    preprocessed_text = preprocess_tweets(text)
    # vectorize the preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # classify the vectorized text using the SVM model
    classification = svmc.predict(vectorized_text)
    # Get sentiment scores
    inputs = tokenizer(text, return_tensors="pt")

    # Make predictions with the model
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    # integrate sentiment analysis with classification
    if classification == 1:  # note: classification==1 for Hate Speech
        if sentiment == "negative":
            return 'Hate Speech detected'
        else:
            return 'No Hate Speech detected'
    else:  # note: classification==0 for Not Hate Speech
        if sentiment == "negative":
            return 'Hate Speech detected'
        else:
            return 'No Hate Speech detected'
##### streamlit code
# Set page title and favicon


st.set_page_config(page_title="Hate Speech Detector")
st.title('Hate Speech detector')
st.write("A hate speech detector is a natural language processing tool that aims to identify and classify text that contains hateful language against an individual or group based on their race, ethnicity, religion, gender, sexual orientation, or other personal characteristics. The detector uses machine learning algorithms to analyze the text, identify patterns, and assign a label of hate speech or not hate speech. The tool can be used to monitor and detect hate speech in social media, online forums, news articles, and other text-based content, with the goal of promoting respectful and inclusive communication online and offline")
CUSTOM_CSS = """
<style>
h1 {
    color: black;
    font-size: 24px;
    text-align: center;
}
</style>
"""

# Define app layout
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# user_input=st.text_input("Type in speech")
user_input = st.text_input("Input text below:")
button=st.button(label="Detect hate speech")

if button and input is not None:
    if user_input != "":
        classification = classify_text(user_input)
        st.write(classification)
    else:
        st.write("Please enter some text.")
st.write('Please input a complete sentence.')