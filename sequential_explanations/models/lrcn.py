import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization, Bidirectional, Reshape
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import logging
from models.keras_callbacks import WeightRestorer
from tensorflow.keras.applications import resnet


# Wrapper class for the LRCN model (CNN + LSTM)
class LRCNClassifier:
    def __init__(self): # Minimal constructor
        self.params = {}

    def initialize(self, exp_params):
        self.params = exp_params
        self.params['model'] = 'lrcn'    # As a marker
        latent_dim = self.params['latent_dim']    # No. of features of LSTM input
        layer_nodes = exp_params['lstm_layer_units']
        num_classes = exp_params['output_nodes']
        time_steps = exp_params['lstm_time_steps']


        # ---------------------------
        # CNN part of the LRCN model

        # Keep the last Global Avg Pooling layer (but not the final Dense [softmax] layer with 1000 nodes for classes)
        cnn_base = resnet.ResNet152(weights='imagenet', pooling='avg', include_top=False)

        # self.cnn_base_with_top = resnet.ResNet152(weights='imagenet', include_top=True)   # For comparison

        for layer in cnn_base.layers:
            layer.trainable = False

        a = cnn_base.output
        cnn_out = Dense(latent_dim, activation=None)(a)

        # self.cnn_full_model = Model(inputs=self.cnn_base.input, outputs=cnn_out)    # Keep aside the CNN full model (unused)


        # ---------------------------
        # LSTM part of the LRCN model

        lstm_layer_in = Reshape((time_steps, latent_dim))(cnn_out)

        for nodes in layer_nodes:
            lstm_layer_in = Bidirectional(LSTM(nodes, return_sequences=True))(lstm_layer_in)

        classifications = TimeDistributed(Dense(num_classes, activation='softmax'))(lstm_layer_in)

        # self.lstm_model = Model(inputs=cnn_out, outputs=classifications)      # Keep aside the LSTM full model (unused)


        # ---------------------------
        # Full LRCN model (CNN + LSTM)

        self.lrcn_model = Model(inputs=cnn_base.input, outputs=classifications)

        self.lrcn_model.compile(optimizer='adam', loss='binary_crossentropy')

        logger = logging.getLogger(__name__)
        # self.lrcn_model.summary(print_fn=logger.info)

        preproc_func = resnet.preprocess_input
        decode_preds_func = resnet.decode_predictions

    def get_tf_model(self):
        return self.lrcn_model

    def set_tf_model(self, model):
        self.lrcn_model = model

    def save(self, filename):
        self.lrcn_model.save(filename, overwrite=True, save_format='h5')

    def load_model(self, filename):
        self.lrcn_model = tf.keras.models.load_model(filename)
