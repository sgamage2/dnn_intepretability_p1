# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:58:41 2020

@author: daisy
"""


import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk_stopw = stopwords.words('english')

def tokenize (text):        #   no punctuation &amp; starts with a letter &amp; between 3-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]

def get20News():
    X, labels, labelToName  = [], [], {}
    twenty_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    for i, article in enumerate(twenty_news['data']):
        stopped = tokenize (article)
        if (len(stopped) == 0):
            continue
        groupIndex = twenty_news['target'][i]
        X.append(stopped)
        labels.append(groupIndex)
        labelToName[groupIndex] = twenty_news['target_names'][groupIndex]
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens

def getEmbeddingMatrix (word_index, vectorSource):
    wordVecSources = {'fasttext' : './vectors/crawl-300d-2M-subword.vec', 'custom-fasttext' : './vectors/' + '20news-fasttext.json' }
    f = open (wordVecSources[vectorSource])
    allWv = {}
    if (vectorSource == 'custom-fasttext'):
        allWv = json.loads(f.read())
    elif (vectorSource == 'fasttext'):
        errorCount = 0
        for line in f:
            values = line.split()
            word = values[0].strip()
            try:
                wv = np.asarray(values[1:], dtype='float32')
                if (len(wv) != wvLength):
                    errorCount = errorCount + 1
                    continue
            except:
                errorCount = errorCount + 1
                continue
            allWv[word] = wv
        print ("# Bad Word Vectors:", errorCount)
    f.close()
    embedding_matrix = np.zeros((len(word_index)+1, wvLength))  # +1 for the masked 0
    for word, i in word_index.items():
        if word in allWv:
            embedding_matrix[i] = allWv[word]
    return embedding_matrix
# The end result is a matrix where each row represents a 300 long vector for a word. 
# The words/rows are ordered as per the integer index in the word_index dictionary – {word:index}. 
# In case of Keras, the words are ordered based on their frequency.
X, labels, labelToName, nTokens = get20News()
print ('Token Summary. min/avg/median/std/85/86/87/88/89/90/91/92/93/94/95/99/max:',)
print (np.amin(nTokens), np.mean(nTokens),np.median(nTokens),np.std(nTokens),np.percentile(nTokens,85),np.percentile(nTokens,86),np.percentile(nTokens,87),np.percentile(nTokens,88),np.percentile(nTokens,89),np.percentile(nTokens,90),np.percentile(nTokens,91),np.percentile(nTokens,92),np.percentile(nTokens,93),np.percentile(nTokens,94),np.percentile(nTokens,95),np.percentile(nTokens,99),np.amax(nTokens))
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)
print ('X, labels #classes classes {} {} {} {}'.format(len(X), str(labels.shape), numClasses, namesInLabelOrder))
# Turn text into 200-long integer sequences, padding with 0 if necessary to maintain the length at 200
import tensorflow as tf
sequenceLength = 200

kTokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=sequenceLength,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
kTokenizer.fit_on_texts(X)
encoded_docs = kTokenizer.texts_to_sequences(X)
Xencoded = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=sequenceLength, padding='post')

print ('Vocab padded_docs {} {}'.format(len(kTokenizer.word_index), str(Xencoded.shape)))
# 107196 is the number of unique words in the corpus
# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]

import os
import time
import string
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import random as rn



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
# Build the corpus and sequences


wvLength = 300 #300 long numerical vector for the corresponding words
vectorSource = str('fasttext') # none, fasttext, custom-fasttext


train_labels = tf.keras.utils.to_categorical(labels[train_indices], len(labelToName))
test_labels = tf.keras.utils.to_categorical(labels[test_indices], len(labelToName))

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', restore_best_weights=False)
model = tf.keras.models.Sequential()
if (vectorSource != 'none'):
    embedding_matrix = getEmbeddingMatrix (kTokenizer.word_index, vectorSource)
    embedding = tf.keras.layers.Embedding(input_dim=len(kTokenizer.word_index)+1, output_dim=wvLength, weights=[embedding_matrix], input_length=sequenceLength, trainable=False, mask_zero=True)
else:
    embedding = tf.keras.layers.Embedding(input_dim=len(kTokenizer.word_index)+1, output_dim=wvLength, input_length=sequenceLength, trainable=True, mask_zero=True)
model.add(embedding)
# model.add(tf.keras.layers.LSTM(units=150, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
start_time = time.time()
result = {}

history = model.fit(x=train_x, y=train_labels, epochs=50, batch_size=32, shuffle=True, validation_data = (test_x, test_labels), verbose=2, callbacks=[early_stop])

result['history'] = history.history

result['test_loss'], result['test_accuracy'] = model.evaluate(test_x, test_labels, verbose=2)

predicted = model.predict(test_x, verbose=2)

predicted_labels = predicted.argmax(axis=1)
result['confusion_matrix'] = confusion_matrix(labels[test_indices], predicted_labels).tolist()
result['classification_report'] = classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)
print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
elapsed_time = time.time() - start_time
print ('Time Taken:', elapsed_time)
result['elapsed_time'] = elapsed_time