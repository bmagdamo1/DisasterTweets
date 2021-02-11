import numpy as np
import pandas as pd
import string
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron


train = pd.read_csv('train.csv')
#stop = stopwords.words('english')
#test = pd.read_csv('test.csv')
target = train['target']
data = train['text']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

def url(text):
    stuff = re.compile(r'https?://\S+|www\.\S+')
    return stuff.sub(r'', text)

def emoji(text):
    stuff = re.compile(
        '['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return stuff.sub(r'', text)

def html(text):
    stuff = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(stuff, '', text)

def punctuation(text):
    stuff = str.maketrans('', '', string.punctuation)
    return text.translate(stuff)

train['text'] = train['text'].apply(lambda x: url(x))
train['text'] = train['text'].apply(lambda x: emoji(x))
train['text'] = train['text'].apply(lambda x: html(x))
train['text'] = train['text'].apply(lambda x: punctuation(x))
train['text'] = train['text'].apply(lambda x: str.lower(x))
#train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


vectorized = Pipeline([('CVec', CountVectorizer(stop_words='english')),
                     ('Tfidf', TfidfTransformer())])

X_train_tranformed = vectorized.fit_transform(X_train)
X_test_tranformed = vectorized.transform(X_test)

classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(n_estimators=500),
    "SVM": SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    'Perceptron': Perceptron(class_weight='balanced')
        }

no_classifiers = len(classifiers.keys())

def batch_classify(X_train_tranformed, y_train, X_test_tranformed, y_test, verbose = True):
    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)), columns = ['Classifier', 'AUC', 'F1 Score'])
    count = 0
    for key, classifier in classifiers.items(): 
        classifier.fit(X_train_tranformed, y_train)
        y_predicted = classifier.predict(X_test_tranformed)
        df_results.loc[count,'Classifier'] = key
        df_results.loc[count,'AUC'] = roc_auc_score(y_test, y_predicted)
        df_results.loc[count,'F1 Score'] = f1_score(y_test, y_predicted)
        count+=1

    return df_results

df_results = batch_classify(X_train_tranformed, y_train,X_test_tranformed, y_test)
print(df_results.sort_values(by='F1 Score', ascending=False))

