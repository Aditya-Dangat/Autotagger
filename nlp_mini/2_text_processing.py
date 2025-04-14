!pip install --upgrade gupload

# from pydrive.auth import GoogleAuth
from google.colab import auth
auth.authenticate_user()

from google.colab import output
output.clear()

!pip install --upgrade --no-cache-dir gdown
output.clear()

"""# Text Processing"""

!gdown 1vwAjp9dJwxyiVdyalYZbEMMH8uIS0oVY
!unzip eda.zip

"""In the notebook, I will focus on data cleansing, such as removing HTML tags from body text, and some other text processing, so that the dataset will be ready for feature extraction."""

import os

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

pd.options.display.max_colwidth = 255
tqdm.pandas()

df = pd.read_pickle(f"{DATA_DIR}/eda.pkl")

df.head()

"""## Understand text length"""

min_title_length = df["title"].str.len().min()
max_title_length = df["title"].str.len().max()
min_body_length = df["body"].str.len().min()
max_body_length = df["body"].str.len().max()

print(f"min_title_length: {min_title_length}")
print(f"max_title_length: {max_title_length}")
print(f"min_body_length: {min_body_length}")
print(f"max_body_length: {max_body_length}")

df[df["title"].str.len() == min_title_length]

df[df["title"].str.len() == max_title_length]

df[df["body"].str.len() == min_body_length]

df[df["body"].str.len() == max_body_length]

"""## Use BeautifulSoup to remove HTML tags from body text"""

df["body"] = df["body"].progress_apply(lambda text: BeautifulSoup(text, "lxml").text)

df.head()

# checkpoint
df.to_pickle(f"{DATA_DIR}/tp-1.pkl")

!gdown 1FtZ-XkQhK6Wn5KDRM2r4kP96U_gbh6SP

!cd data && unzip tp-1.zip

df = pd.read_pickle(f"{DATA_DIR}/tp-1.pkl")

"""## Lower case, remove newline and punctuations; tokenize and handle symbols in topics"""

!gdown 1FtZ-XkQhK6Wn5KDRM2r4kP96U_gbh6SP
!unzip tp-1.zip

df = pd.read_pickle(f"{DATA_DIR}/tp-1.pkl")

df["body"] = df["body"].str.lower()

len(df.index)

df=df[:321521]

import nltk
nltk.download("punkt")

# we have to keep a list of topics with symbols or digits that people will actually type in because of how nltk handles word tokenization
# this list includes tags that have more than 10,000 questions as of 2020 Jan
topics_with_symbols = ["c#", "c++", ".net", "asp.net", "node.js", "objective-c", "unity3d", "html5", "css3", \
                       "d3.js", "utf-8", "neo4j", "scikit-learn", "f#", "3d", "x86"]

df["body_tokenized"] = df["body"].progress_apply(lambda text: [word for word in nltk.word_tokenize(text) \
                                                               if word.isalpha() or word in list("+#") + topics_with_symbols])

df.to_pickle(f"{DATA_DIR}/tp-1.5.pkl")

!zip tp-1.5.zip data/tp-1.5.pkl

# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'tp-1.5.zip'

# retokenize topics including meaningful symbols such as C#, C++
# Multi-Word Expression Tokenizer
mwe_tokenizer = nltk.MWETokenizer(separator="")
mwe_tokenizer.add_mwe(("c", "#"))
mwe_tokenizer.add_mwe(("c", "+", "+"))
mwe_tokenizer.add_mwe(("f", "#"))

df["body_tokenized"] = df["body_tokenized"].progress_apply(lambda tokens: [token for token in mwe_tokenizer.tokenize(tokens)])

"""## Remove stop words"""

from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def filter_stop_words(words):
    words_filtered = []
    for word in words:
        if word not in stop_words:
            words_filtered.append(word)
    return words_filtered

df["body_tokenized"] = df["body_tokenized"].progress_apply(filter_stop_words)

"""Let's take a look at the results:"""

df.sample(5)

"""## Repeat the steps above on title column as well mwe_tokenizer"""

df["title"] = df["title"].str.lower()

df["title_tokenized"] = df["title"].progress_apply(lambda text: [word for word in nltk.word_tokenize(text) \
                                                               if word.isalpha() or word in list("+#") + topics_with_symbols])

df["title_tokenized"] = df["title_tokenized"].progress_apply(lambda tokens: [token for token in mwe_tokenizer.tokenize(tokens)])

df["title_tokenized"] = df["title_tokenized"].progress_apply(filter_stop_words)

df.sample(5)

# checkpoint
df.rename(columns={"tag": "tags"}, inplace=True)
df[["id", "title_tokenized", "body_tokenized", "tags"]].to_pickle(f"{DATA_DIR}/tp-2.pkl")

df.to_pickle(f"{DATA_DIR}/tp-2.pkl")

!zip tp-2.zip data/tp-2.pkl

# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'tp-2.zip'