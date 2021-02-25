import numpy as np
import pandas as pd
import nltk
import utils
import torch
import optuna
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
from sklearn import model_selection

LOGISTIC_REGRESSION = "Logistic Regression"
RANDOM_FOREST = "Random Forest"
ADA_BOOST = "AdaBoost"
GRADIENT_BOOSTING_CLASSIFIER = "Gradient Boosting Classifier"
NEURAL_NETWORK = "Neural Network"

train = pd.read_csv('./Data/Raw/train.csv')
target = train['target']
data = train['text']
np.random.seed(19970901)

# Preprocess the text data
print("Preprocessing...")
train['text'] = train['text'].apply(lambda x: utils.url(x))
train['text'] = train['text'].apply(lambda x: utils.emoji(x))
train['text'] = train['text'].apply(lambda x: utils.html(x))
train['text'] = train['text'].apply(lambda x: utils.punctuation(x))
train['text'] = train['text'].apply(lambda x: str.lower(x))
#train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Setup DistilliBERT from *******
print("Setting up DistilliBERT...")
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Convert the raw text into DistilliBERT vector encodings to the text 
print("Encoding...")
train['encoding'] = train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padd the distilliBERT encodings for the sake of the 
print("Padding...")
max_tokenized_length = max([ len(l) for l in train['encoding'].ravel() ])
padded = np.array([i + [0]*(max_tokenized_length-len(i)) for i in train['encoding'].values])
attention_mask = np.where(padded != 0, 1, 0)

print("Creating tensor mask...")
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

print("Figuring out hidden states...")
with torch.no_grad():
    print("\tone")
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    print("\ttwo")
print("Parsing hidden state results...")
features = last_hidden_states[0][:,0,:].numpy()
train_features, test_features, train_labels, test_labels = train_test_split(features, target)

print("Defining classifiers...")
classifiers = {
    LOGISTIC_REGRESSION: LogisticRegression(class_weight='balanced'),
    RANDOM_FOREST: RandomForestClassifier(),
    ADA_BOOST: AdaBoostClassifier(n_estimators=500),
    GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier(),
    NEURAL_NETWORK: MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(100,1000,20))
    # "SVM": SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
    # 'Perceptron': Perceptron(class_weight='balanced'),
}

'''
num_classifiers = len(classifiers.keys())
def batch_classify(X_train_transformed, y_train, X_test_transformed, y_test, verbose = True):
    df_results = pd.DataFrame(data=np.zeros(shape=(num_classifiers,4)), columns = ['Classifier', 'AUC', 'Accuracy', 'F1 Score'])
    count = 0
    for key, classifier in classifiers.items():
        classifier.fit(X_train_transformed, y_train)
        y_predicted = classifier.predict(X_test_transformed)
        df_results.loc[count,'Classifier'] = key
        df_results.loc[count,'AUC'] = roc_auc_score(y_test, y_predicted)
        df_results.loc[count,'Accuracy'] = accuracy_score(y_test, y_predicted)
        df_results.loc[count,'F1 Score'] = f1_score(y_test, y_predicted)
        count+=1

    return df_results
'''

# df_results = batch_classify(train_features, train_labels, test_features, test_labels)
# print(df_results.sort_values(by='F1 Score', ascending=False))

#Step 1. Define an objective function to be maximized.
def objective(trial):

    print("In objective (trial=" + str(trial) + ")")

    classifier_name = trial.suggest_categorical("classifier", [LOGISTIC_REGRESSION, RANDOM_FOREST]) # , ADA_BOOST, GRADIENT_BOOSTING_CLASSIFIER, NEURAL_NETWORK])
    
    # Step 2. Setup values for the hyperparameters:
    if classifier_name == LOGISTIC_REGRESSION:
        print("\tLogistic Regression")
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        # classifier_obj = classifiers[LOGISTIC_REGRESSION].LogisticRegression(C=logreg_c)
        classifier_obj = LogisticRegression(C=logreg_c)
    elif RANDOM_FOREST:
        print("\tRandom Forest")
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators
        )
        # classifier_obj = classifiers[RANDOM_FOREST].RandomForestClassifier(
        #     max_depth=rf_max_depth, n_estimators=rf_n_estimators
        # )
    elif ADA_BOOST:
        print("Not doing anything for AdaBoost")
        return
    elif GRADIENT_BOOSTING_CLASSIFIER:
        print("Not doing anything for Gradient Boosting Classifier")
        return
    elif NEURAL_NETWORK:
        print("Not doing anything for Neural Network")
        return   
    else:
        print("Not doing anything for (default)")
        return

    # Step 3: Scoring method:
    print("\tScoring trial...")
    score = model_selection.cross_val_score(classifier_obj, train_features, train_labels, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

# Step 4: Running it
print("Running Study...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)