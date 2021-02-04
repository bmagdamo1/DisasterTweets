import numpy as np
import pandas as pd
import string
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train = pd.read_csv('train.csv')
stop = stopwords.words('english')

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
print(stop)
#print(train['text'][0])