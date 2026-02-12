import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# 1) Load dataset
# ==============================

df = pd.read_csv("train/train.csv")

# ==============================
# 2) Create Binary Sentiment
# ==============================

label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# 1 = Negative (any toxic label)
# 0 = Positive (no toxic labels)
df["sentiment"] = (df[label_columns].sum(axis=1) > 0).astype(int)

# ==============================
# 3) Features & Target
# ==============================

X = df["comment_text"].astype(str)
y = df["sentiment"]

# ==============================
# 4) Train / Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5) Build Model Pipeline
# ==============================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )),
    ("model", LogisticRegression(
        max_iter=1000
    ))
])

# ==============================
# 6) Train Model
# ==============================

pipeline.fit(X_train, y_train)

# ==============================
# 7) Save Model
# ==============================

joblib.dump(pipeline, "sentiment_model.pkl")

print("Model trained and saved successfully!")
print("Class distribution:")
print(df["sentiment"].value_counts())
