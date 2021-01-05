import json_lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from googletrans import Translator, constants
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.naive_bayes import MultinomialNB
from cross_validation import CrossValidate
from sklearn.neural_network import MLPClassifier
from evaluate import Evaluate

evaluator = Evaluate()
cross_validation = CrossValidate()

nltk.download('punkt')
nltk.download('stopwords')


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# ------- Get data -------
X = []
y = []
z = []

with open("../data/new_data.jl", "rb") as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item["voted_up"])
        z.append(item["early_access"])


# --- Process the text, vectorize, tokenize, stemmers, etc.
def process_data(X_value):
    translator = Translator()
    tokenizer = CountVectorizer().build_tokenizer()
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
        # tokenizer=LemmaTokenizer(),
        #     strip_accents='ascii',
        #     min_df=0.0005,
        # sublinear_tf=True,
        max_df=0.8,
    )
    # vectorizer = CountVectorizer(
    # tokenizer=LemmaTokenizer(),
    # lowercase=True,
    # max_df=0.5,
    # min_df=10,
    # stop_words=nltk.corpus.stopwords.words('english'))
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
k = 25

kf = KFold(n_splits=k)
print("KFOLD: ", k)
for train, test in kf.split(X):
    print("-> ")
    model = LogisticRegression(random_state=0)
    model.fit(X[train], z[train])
    predictions = model.predict(X[test])
    preds.extend(predictions)

evaluator.calculate_confusion_matrix(y, preds, "Logistic Regression")
evaluator.plot_roc_curve(y, preds, "Classifier KFold = 50")
