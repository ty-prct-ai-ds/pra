from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


categories = None
newsgroups = fetch_20newsgroups(
    subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))


X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.3, random_state=42
)

model = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    MultinomialNB()
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy:{accuracy_score(y_test, y_pred)}")
print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

def show_top_keywords(model, categories, n=10):
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['multinomialnb']
    feature_names = np.array(vectorizer.get_feature_names_out())

    for i, category in enumerate(categories):
        top_indices = np.argsort(classifier.feature_log_prob_[i])[-n:]
        print(f"Top patterns fo category '{category}'")
        print(", ".join(feature_names[top_indices]))


show_top_keywords(model, newsgroups.target_names, n=8)
