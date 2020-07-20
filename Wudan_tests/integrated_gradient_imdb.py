# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:19:21 2020

@author: daisy
"""


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import os
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())

max_features = 10000  # max number of words
maxlen = 100          # truncated sentence length = 100

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
test_labels = y_test.copy()
train_labels = y_train.copy()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])

def decode_sentence(x, reverse_index):
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, 'UNK') for i in x])

print(decode_sentence(x_test[1], reverse_index))

batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250

load_model = False
save_model = True

filepath = './model_imdb/'  # change to directory where model is downloaded
if load_model:
    model = tf.keras.models.load_model(os.path.join(filepath, 'model.h5'))
else:
    print('Build model...')
    
    inputs = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(max_features,
                                   embedding_dims)(inputs)
    out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedded_sequences)
    out = layers.Bidirectional(layers.LSTM(64))(out)
    outputs = Dense(2, activation='softmax')(out)
        
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=3,
              validation_data=(x_test, y_test))
    if save_model:  
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        model.save(os.path.join(filepath, 'model.h5'))
        
print(model.summary())

n_steps = 50
method = "gausslegendre"
internal_batch_size = 100
nb_samples = 10
ig  = IntegratedGradients(model,
                          layer=model.layers[1],
                          n_steps=n_steps, 
                          method=method,
                          internal_batch_size=internal_batch_size)

x_test_sample = x_test[:nb_samples]
predictions = model(x_test_sample).numpy().argmax(axis=1)
explanation = ig.explain(x_test_sample, 
                         baselines=None, 
                         target=predictions)


print(explanation.meta)
# Data fields from the explanation object
print(explanation.data.keys())
# Get attributions values from the explanation object
attrs = explanation.attributions
print('Attributions shape:', attrs.shape)
attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)
i = 5
x_i = x_test_sample[i]
attrs_i = attrs[i]
pred = predictions[i]
pred_dict = {1: 'Positive review', 0: 'Negative review'}
print('Predicted label =  {}: {}'.format(pred, pred_dict[pred]))
from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return "<mark style=background-color:{}>{} </mark>".format(color, string)
def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)
    
    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors
words = decode_sentence(x_i, reverse_index).split()
colors = colorize(attrs_i)
HTML("".join(list(map(hlstr, words, colors))))
