import json_lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import re
from sklearn.naive_bayes import MultinomialNB
from cross_validation import CrossValidate
from evaluate import Evaluate

evaluator = Evaluate()
cross_validation = CrossValidate()

nltk.download('punkt')
nltk.download('stopwords')

# ------- Get data -------
X = []
y = []
z = []

with open("../data/balanced_early_access.jl", "rb") as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item["voted_up"])
        z.append(item["early_access"])


# --- Process the text, vectorize, tokenize, stemmers, etc.
def process_data(X_value):
    tokenizer = TfidfVectorizer().build_tokenizer()
    stemmer = PorterStemmer()
    total_x = []

    for text in X_value:
        processed_text = tokenizer(text)

        processed_text = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token) for token in processed_text]
        processed_text = [
            re.sub("(@[A-Za-z0-9_]+)", "", token) for token in processed_text
        ]
        processed_text = [stemmer.stem(token) for token in processed_text]
        processed_text = " ".join(processed_text)
        total_x.append(processed_text)

    vectorizer = TfidfVectorizer(
        stop_words=nltk.corpus.stopwords.words('english'),
        max_df=0.8,
    )

    X_vectorized = vectorizer.fit_transform(total_x)
    X_vectorized = X_vectorized.toarray()
    return X_vectorized


# --- Now good data ---
X = process_data(X)
z = np.array(z) * 1

# cross_validation.do_cross_validation_kfold(X, y)
# cross_validation.do_cross_validation_c(X, y)
# cross_validation.do_cross_validation_knn(X, y)
# exit(1)
print("--- Begin model training ---")

plt.figure(1)
preds = []
k = 100

kf = KFold(n_splits=k)
print("KFOLD: ", k)
for train, test in kf.split(X):
    print("-> ")
    model = LogisticRegression()
    model.fit(X[train], z[train])
    predictions = model.predict(X[test])
    preds.extend(predictions)

evaluator.calculate_confusion_matrix(z, preds,
                                     "Logistic Regression, Equalised Data")
evaluator.plot_roc_curve(z, preds,
                         "Logistic Regression, Equalised Data, KFold = 100")
