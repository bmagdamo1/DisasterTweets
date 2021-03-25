# !pip3 install pandas
# !pip3 install numpy
# !pip3 install simpletransformers
# !pip3 install wandb

import wandb
import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import numpy as np
import string
import re
import nltk
import wandb
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv('Data/Processed/cleaned_train.csv')
train_df = prepareDataframe(train_df)
train_df.info()

test_df = pd.read_csv('Data/Processed/cleaned_test.csv')
test_df = prepareDataframe(test_df)
test_df.info()

#preprocessing
train_df.drop(['keyword', 'location', 'id'], axis=1, inplace=True)
stop = stopwords.words('english')
np.random.seed(19970901)

#specify simple transformer model with number of epochs
model_args = ClassificationArgs(num_train_epochs = 1,learning_rate = 3e-5)
model = ClassificationModel("roberta", "roberta-base", args=model_args, use_cuda=True)

#Train model and evaluate test set
model.train_model(train_df)

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
}

sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Initialize a new wandb run
wandb.init()

# Train the model
# model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
model.eval_model(test_df)

# Sync wandb
wandb.join()

wandb.agent(sweep_id)

'''
!! Utility Functions !!
'''

def prepareDataframe(df):
    train_df['text'] = train_df['text'].apply(lambda x: url(x))
    train_df['text'] = train_df['text'].apply(lambda x: html(x))
    train_df['text'] = train_df['text'].apply(lambda x: punctuation(x))
    train_df['text'] = train_df['text'].apply(lambda x: str.lower(x))

def url(text):
    stuff = re.compile(r'https?://\S+|www\.\S+')
    return stuff.sub(r'', text)

def html(text):
    stuff = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(stuff, '', text)

def punctuation(text):
    stuff = str.maketrans('', '', string.punctuation)
    return text.translate(stuff)
