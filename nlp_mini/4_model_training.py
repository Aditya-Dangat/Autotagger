# -*- coding: utf-8 -*-

!pip install --upgrade gupload

# from pydrive.auth import GoogleAuth
from google.colab import auth
auth.authenticate_user()

from google.colab import output
output.clear()

!pip install --upgrade --no-cache-dir gdown
output.clear()

"""# Model Training"""

!gdown 1b7eNpt2JiGSey-2WLxtKr46iiTWoWDJb
!unzip train_test.zip

"""## Model training"""

import os
import joblib

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")
MODEL_DIR = os.path.join(HOME_DIR, "model")

import pandas as pd
from tqdm import tqdm

pd.options.display.max_colwidth = 255
tqdm.pandas()

X_train = joblib.load(f"{DATA_DIR}/x_train.pkl")
X_test = joblib.load(f"{DATA_DIR}/x_test.pkl")
y_train = joblib.load(f"{DATA_DIR}/y_train.pkl")
y_test = joblib.load(f"{DATA_DIR}/y_test.pkl")
y_classes = joblib.load(f"{DATA_DIR}/y_classes.pkl")

# %%time

# from sklearn.multiclass import OneVsRestClassifier
# from xgboost import XGBClassifier

# xgb_classifier = XGBClassifier(max_depth=5,
#                                eta=0.2,
#                                gamma=4,
#                                min_child_weight=6,
#                                subsample=0.8,
#                                early_stopping_rounds=10,
#                                num_round=200,
#                                n_jobs=-1)

# clf = OneVsRestClassifier(xgb_classifier)
# clf.fit(X_train, y_train)

"""`XGBClassifier` takes way too long to train, so we switch to linear model using `SGDClassifier`."""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import SGDClassifier
# 
# sgd_classifier = SGDClassifier(n_jobs=-1)
# 
# clf = OneVsRestClassifier(sgd_classifier)
# clf.fit(X_train, y_train)

# save the model
joblib.dump(clf, f"{MODEL_DIR}/one_vs_rest_classifier.pkl")

!zip one_vs_rest_classifier.zip model/one_vs_rest_classifier.pkl
# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'one_vs_rest_classifier.zip'

clf = joblib.load(f"{MODEL_DIR}/one_vs_rest_classifier.pkl")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# y_pred = clf.predict(X_test)

X_test

y_pred

joblib.dump(y_pred, f"{MODEL_DIR}/y_pred.pkl")

!zip y_pred.zip model/y_pred.pkl
# Upload file to Drive
!gupload --to '1-rs3VAKRW4-LDuJn-09egK0oXXzPm6ni' 'y_pred.zip'

"""## Model evaluation"""

!gdown 1kT7xlqCfYl49NSqnN7jYLV8XaNCzw9kL
!unzip one_vs_rest_classifier.zip
!gdown 1DOvZj4LKUSLanej7RZdGhlP3xOtE2TXD
!unzip y_pred.zip

clf = joblib.load(f"{MODEL_DIR}/one_vs_rest_classifier.pkl")

y_pred = joblib.load(f"{MODEL_DIR}/y_pred.pkl")

"""### Precision, Recall, F-1 score"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# from sklearn.metrics import precision_recall_fscore_support as score
# 
# precision, recall, fscore, support = score(y_test, y_pred)
# 
# print(f"precision: {precision}")
# print(f"recall: {recall}")
# print(f"fscore: {fscore}")
# print(f"support: {support}")

"""### Hamming loss"""

from sklearn.metrics import hamming_loss

hamming = []

for i, (test, pred) in enumerate(zip(y_test.T, y_pred.T)):
    hamming.append(hamming_loss(test, pred))

metric_df = pd.DataFrame(data=[precision, recall, fscore, hamming, support],
                         index=["Precision", "Recall", "F-1 score", "Hamming loss", "True count"],
                         columns=y_classes)

metric_df

metric_df.loc[:, metric_df.columns.str.startswith(".net")]

"""Let's take a look at the top 10 tags:"""

top_ten_tags = ["javascript", "java", "c#", "php", "android", "jquery", "python", "html", "c++", "ios"]
metric_df[top_ten_tags]

import numpy as np
metric_df[top_ten_tags].apply(np.mean, axis=1)

non_zero_metric_df = metric_df.loc[:, metric_df.loc["F-1 score"] > 0]

non_zero_metric_df.apply(np.mean, axis=1)