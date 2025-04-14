import re

import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
from scipy.sparse import hstack

tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

X_title_vect = joblib.load(f"./model/X_title_vect.pkl")
X_body_vect = joblib.load(f"./model/X_body_vect.pkl")
y_bin = joblib.load(f"./model/y_bin.pkl")
clf = joblib.load(f"./model/clf.pkl")

def remove_html(text):
    # Remove html and convert to lowercase
    return re.sub(r"\<[^\>]\>", "", text).lower()

def remove_stopwords(text):    
    # tokenize the text
    words = tokenizer.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def remove_punc(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    
    # remove punctuations from each token
    tokens = list(map(lambda token: re.sub(r"[^A-Za-z0-9]+", " ", token).strip(), tokens))
    
    # remove empty strings from tokens
    tokens = list(filter(lambda token: token, tokens))
    return ' '.join(map(str, tokens))

def stem_text(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    
    # stem each token
    tokens = list(map(lambda token: stemmer.stem(token), tokens))
    return ' '.join(map(str, tokens))

# preprocessing title and body
def preprocess_text(text):
    text = remove_html(text)
    text = remove_stopwords(text)
    text = remove_punc(text)
    text = stem_text(text)
    return text

def predict_tags(q_title, q_body):
    q_title = preprocess_text(q_title)
    q_body = preprocess_text(q_body)

    X_app_title = X_title_vect.transform([q_title])
    X_app_title = 2 * X_app_title
    X_app_body = X_body_vect.transform([q_body])
    X_app = hstack([X_app_title, X_app_body])

    y_app = clf.predict(X_app)
    print(y_app)

    y_out = y_bin.inverse_transform(y_app)
    print(y_out)
    return y_out[0]

# Actual Application
st.write("""
# Tagging Stack Overflow Questions
#### This app predicts tags for Stack Overflow questions!
""")
st.write('### NLP Project done by:')
st.markdown("<h3 style='text-align: right; font-family: Source Sans Pro, sans-serif;'>Aditya D. | Sanika J. | Anish L. | Latikesh K.</h3>", unsafe_allow_html=True)

#st.sidebar.header('User Input Parameters')

def user_input_features():
    #q_body1 = st.sidebar.slider('Question Body', 2.0, 4.4, 3.4)
    q_title = st.text_input('Question Title', value="", on_change=None, placeholder=None)
    q_body = st.text_area('Question Body', value="", height=None, on_change=None, placeholder=None)
    return q_title, q_body

q_title, q_body = user_input_features()
tags_out = predict_tags(q_title, q_body)

st.write("# Answer")
st.write(f"#### {tags_out}")

