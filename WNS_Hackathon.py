# Generated from: WNS_Hackathon.ipynb
# Converted at: 2025-12-18T18:02:15.191Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")


train.head()
train.info()


target = 'is_promoted'


X = train.drop(columns=[target])
y = train[target]


cat_cols = X.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.fit_transform(test[col])


X.fillna(X.median(), inplace=True)
test.fillna(test.median(), inplace=True)


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # VERY important for F1
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',  # IMPORTANT for F1
    random_state=42
)

model.fit(X_train, y_train)


val_pred = model.predict(X_val)
f1_score(y_val, val_pred)


model.fit(X, y)


test_pred = model.predict(test)


sample_sub[target] = test_pred
sample_sub.to_csv("submission.csv", index=False)

sample_sub.head()