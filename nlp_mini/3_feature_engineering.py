
!pip install --upgrade gupload

# from pydrive.auth import GoogleAuth
from google.colab import auth
auth.authenticate_user()

from google.colab import output
output.clear()

!pip install --upgrade --no-cache-dir gdown
output.clear()

"""# Feature Engineering"""

!gdown 1mR4iyGJhL7pewcHXwyJTnmbaG9khA2zl
!unzip tp-2.zip

"""In this notebook, I will focus on feature extraction from the two text columns, i.e. title and body, so that the data set will be ready for model training."""

import os

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")

import pandas as pd
from tqdm import tqdm

pd.options.display.max_colwidth = 255
tqdm.pandas()

df = pd.read_pickle(f"{DATA_DIR}/tp-2.pkl")

df.sample(5)

"""## Number of tags (i.e. classes)"""

from collections import Counter

tag_count = Counter()

def count_tag(tags):
    for tag in tags:
        tag_count[tag] += 1

df["tags"].apply(count_tag)

len(tag_count.values())


most_common_tags = [count[0] for count in tag_count.most_common(4000)]
df["tags"] = df["tags"].progress_apply(lambda tags: [tag for tag in tags if tag in most_common_tags])

df[df["tags"].map(lambda tags: len(tags) > 0)].shape

print(f"Only {321521 - 319855:,} rows of data will be dropped while number of classes is reduced from {len(tag_count.values()):,} to 4,000.")

df = df[df["tags"].map(lambda tags: len(tags) > 0)]

# checkpoint
df.to_pickle(f"{DATA_DIR}/fe-1.pkl")

!zip fe-1.zip data/fe-1.pkl

# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'fe-1.zip'

"""## tf-idf"""

!gdown 1idvcc4Bm840Bm6ydlpHCn6Ik08xbC-Pw
!unzip fe-1.zip

df = pd.read_pickle(f"{DATA_DIR}/fe-1.pkl")

from sklearn.feature_extraction.text import TfidfVectorizer
# Term Frequency (tf) Inverse Document Frequency (idf)

# we have already tokenize the text so we need a dummy one to bypass tokenization
def dummy_tokenizer(string): return string

# we will only get the 10,000 most common words for title to limit size of dataset
title_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, lowercase=False, max_features=10000)
title_vectorizer_fit = title_vectorizer.fit(df["title_tokenized"])
x_title = title_vectorizer_fit.transform(df["title_tokenized"])

import joblib

joblib.dump(title_vectorizer_fit, f"{DATA_DIR}/title_vectorizer_fit.pkl")
!zip title_vectorizer_fit.zip data/title_vectorizer_fit.pkl
# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'title_vectorizer_fit.zip'

# we will get the 100,000 most common words for body
body_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, lowercase=False, max_features=100000)
body_vectorizer_fit = body_vectorizer.fit(df["body_tokenized"])
x_body = body_vectorizer_fit.transform(df["body_tokenized"])

joblib.dump(body_vectorizer_fit, f"{DATA_DIR}/body_vectorizer_fit.pkl")
!zip body_vectorizer_fit.zip data/body_vectorizer_fit.pkl
# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'body_vectorizer_fit.zip'

"""Example:"""

df.iloc[[10]]

pd.DataFrame(x_title[:11].toarray(), columns=title_vectorizer.get_feature_names()) \
  .iloc[10].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

pd.DataFrame(x_body[:11].toarray(), columns=body_vectorizer.get_feature_names()) \
  .iloc[10].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

"""It's not that bad, as we can see keywords from the feature like `connect`, `loop`, `c#` and `database`, which are similar to the actual tags.

## Concantenate dataset and train test split
"""

# there is a problematic tag named "nan" which causes string comparison error
df["tags"] = df["tags"].apply(lambda tags: [tag if not isinstance(tag, float) else "nan" for tag in tags])

# give a weight of 2 to title as it should contain more important words than body
x_title = x_title * 2

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

X = hstack([x_title, x_body])
y = df[["tags"]]

from sklearn.preprocessing import MultiLabelBinarizer

multi_label_binarizer = MultiLabelBinarizer(sparse_output=True)
multi_label_binarizer_fit = multi_label_binarizer.fit(y["tags"])
y = multi_label_binarizer_fit.transform(y["tags"])

joblib.dump(body_vectorizer_fit, f"{DATA_DIR}/multi_label_binarizer_fit.pkl")
!zip multi_label_binarizer_fit.zip data/multi_label_binarizer_fit.pkl
# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'multi_label_binarizer_fit.zip'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# checkpoint
import joblib

joblib.dump(X_train, f"{DATA_DIR}/x_train.pkl")
joblib.dump(X_test, f"{DATA_DIR}/x_test.pkl")
joblib.dump(y_train, f"{DATA_DIR}/y_train.pkl")
joblib.dump(y_test, f"{DATA_DIR}/y_test.pkl")
joblib.dump(multi_label_binarizer.classes_, f"{DATA_DIR}/y_classes.pkl")

!zip -r train_test.zip data

# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'train_test.zip'