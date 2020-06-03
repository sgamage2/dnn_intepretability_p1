import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import logging
from models.keras_callbacks import WeightRestorer


# Wrapper class for the LRCN model (CNN + LSTM)
class LRCNClassifier:
    def __init__(self): # Minimal constructor
        self.params = {}

    def initialize(self, exp_params):
        self.params = exp_params
        self.params['model'] = 'lrcn'    # As a marker

        self.lrcn = Sequential()
        self.lrcn.add(resnet152_part)
        self.lrcn.add(lstm_part)