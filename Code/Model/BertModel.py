import numpy as np
import pandas as pd
import nltk
import utils
import torch
import transformers as ppb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('./Data/Raw/train.csv')
#stop = stopwords.words('english')
#test = pd.read_csv('test.csv')
target = train['target']
data = train['text']
np.random.seed(19970901)

# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

# Preprocess the text data
train['text'] = train['text'].apply(lambda x: utils.url(x))
train['text'] = train['text'].apply(lambda x: utils.emoji(x))
train['text'] = train['text'].apply(lambda x: utils.html(x))
train['text'] = train['text'].apply(lambda x: utils.punctuation(x))
train['text'] = train['text'].apply(lambda x: str.lower(x))
#train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# Setup DistilliBERT from *******
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Convert the raw text into DistilliBERT vector encodings to the text 
train['encoding'] = train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padd the distilliBERT encodings for the sake of the 
max_tokenized_length = max([ len(l) for l in train['encoding'].ravel() ])
padded = np.array([i + [0]*(max_tokenized_length-len(i)) for i in train['encoding'].values])
attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

print("Padding and masking complete")


with torch.no_grad():

    print("in here...")
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    print("did the thing")

print("out of there!!")

features = last_hidden_states[0][:,0,:].numpy()

print("Features and labels extracted")

train_features, test_features, train_labels, test_labels = train_test_split(features, target)

print("Split testing and training data")


lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
print(lr_clf.score(test_features, test_labels))

results = pd.DataFrame(data=np.zeros(shape=(1,4)), columns = ['Classifier', 'AUC', 'Accuracy', 'F1 Score'])
y_predicted = lr_clf.predict(test_features)

count = 0
results.loc[count,'Classifier'] = 'LogisticRegression()'
results.loc[count,'AUC'] = roc_auc_score(test_labels, y_predicted)
results.loc[count,'Accuracy'] = accuracy_score(test_labels, y_predicted)
results.loc[count,'F1 Score'] = f1_score(test_labels, y_predicted)
# print(encodings)
#print(train)

print(results)