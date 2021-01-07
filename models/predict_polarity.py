import json_lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ------- GET DATA -------
X = []
y = []
z = []

with open("../data/new_data.jl", "rb") as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item["voted_up"])
        z.append(item["early_access"])


# --- TEXT PROCESSING ---
def process_data(X_value):
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

    return total_x


X = process_data(X)
y = np.array(y) * 1

# --- CROSS VALIDATION ---

# cross_validation.do_cross_validation_max_df(X, y)
# exit(1)

vectorizer = TfidfVectorizer(
    stop_words=nltk.corpus.stopwords.words('english'),
    max_df=0.8,
)

X = vectorizer.fit_transform(X).toarray()
# cross_validation.do_cross_validation_knn(X, y)
# cross_validation.do_cross_validation_kfold(X, y)
# cross_validation.do_cross_validation_c(X, y)
# exit(1)

# --- MODEL TRAINING ---
plt.figure(1)
preds = []
k = 100
kf = KFold(n_splits=k)
print("KFOLD: ", k)
count = 0
for train, test in kf.split(X):
    print("-> ", count)
    count = count + 1
    model = LogisticRegression()
    model.fit(X[train], y[train])
    predictions = model.predict(X[test])
    preds.extend(predictions)

# --- EVALUATION ---
evaluator.calculate_confusion_matrix(y, preds, "-><-")
evaluator.plot_roc_curve(y, preds, "KNN KFold = 100, Neighbours = 40")
