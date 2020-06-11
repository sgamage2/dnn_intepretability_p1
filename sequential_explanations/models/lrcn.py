import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization, Bidirectional, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import logging
from models.keras_callbacks import WeightRestorer
# from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import inception_v3


# Wrapper class for the LRCN model (CNN + LSTM)
class LRCNClassifier:
    def __init__(self): # Minimal constructor
        self.params = {}

    def initialize(self, exp_params):
        self.params = exp_params
        self.params['model'] = 'lrcn'    # As a marker

        self.cnn = self.create_cnn(exp_params)
        self.lstm = self.create_lstm(exp_params)
        self.lrcn_model = self.create_lrcn(exp_params)

        # ---------------------------
        # Full LRCN model (CNN + LSTM)

        # self.lrcn_model = Model(inputs=cnn_base.input, outputs=classifications)

        # self.lrcn_model.compile(optimizer='adam', loss='binary_crossentropy')

        logger = logging.getLogger(__name__)
        self.lrcn_model.summary(print_fn=logger.info)

        # preproc_func = resnet.preprocess_input
        # decode_preds_func = resnet.decode_predictions

    def create_cnn(self, params):
        # Keep the last Global Avg Pooling layer (but not the final Dense [softmax] layer with 1000 nodes for classes)
        cnn_base = inception_v3.InceptionV3(weights='imagenet', pooling='avg', include_top=False)

        # self.cnn_base_with_top = resnet.ResNet152(weights='imagenet', include_top=True)   # For comparison

        for layer in cnn_base.layers:
            layer.trainable = False

        # a = cnn_base.output
        # cnn_out = Dense(latent_dim, activation=None)(a)
        cnn_out = cnn_base.output

        cnn_model = Model(inputs=cnn_base.input, outputs=cnn_out)

        return cnn_model

    def create_lstm(self, params):
        latent_dim = params['latent_dim']  # No. of features of LSTM input
        layer_nodes = params['lstm_layer_units']
        num_classes = params['output_nodes']
        time_steps = params['lstm_time_steps']
        dropouts = params['lstm_layer_dropout_rates']

        lstm = Sequential()

        ret_seq = False if len(layer_nodes) == 1 else True  # To handle only 1 hidden layer

        # First hidden layer (need to specify input layer nodes here)
        lstm.add(LSTM(units=layer_nodes[0], return_sequences=ret_seq, input_shape=(time_steps, latent_dim)))
        lstm.add(Dropout(dropouts[0]))

        # Other hidden layers
        for i in range(1, len(layer_nodes)):
            ret_seq = False if i == len(layer_nodes) - 1 else True    # Last hidden layer should not return sequences
            lstm.add(Bidirectional(LSTM(units=layer_nodes[i], return_sequences=ret_seq)))
            lstm.add(Dropout(dropouts[i]))

        lstm.add(Dense(num_classes, activation='softmax'))

        lstm.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy','top_k_categorical_accuracy'])

        return lstm

    def create_lrcn(self, params):
        time_steps = params['lstm_time_steps']
        width, height, channels = params['image_shape']

        cnn_input = Input(shape=(time_steps, width, height, channels))  # input shape = (num_samples, time_steps, w, h, channels)

        cnn_model = TimeDistributed(Model(inputs=self.cnn.input, outputs=self.cnn.output))
        lstm_model = Model(inputs=self.lstm.input, outputs=self.lstm.output)

        cnn_output = cnn_model(cnn_input)
        lstm_output = lstm_model(cnn_output)

        lrcn_model = Model(cnn_input, lstm_output)

        lrcn_model.compile(optimizer='adam', loss='categorical_crossentropy')

        return lrcn_model

    def fit_lstm(self, X_train, y_train, X_valid=None, y_valid=None):
        epochs = self.params['lstm_epochs']
        batch_size = self.params['lstm_batch_size']
        early_stop_patience = self.params['lstm_early_stop_patience']


        # --------------------------------
        # Create callbacks

        # tensorboard_cb = TensorBoard(log_dir=self.params['tensorboard_log_dir'], batch_size=batch_size)
        # callbacks = [tensorboard_cb]
        callbacks = []

        # Early stopping is enabled
        if X_valid is not None and early_stop_patience > 0 and early_stop_patience < epochs:
            early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                              verbose=2, mode='auto', restore_best_weights=False)
            callbacks.append(early_stopping_cb)

        if X_valid is not None:
            weight_restorer_cb = WeightRestorer(epochs)
            callbacks.append(weight_restorer_cb)

        # --------------------------------
        # Fit

        if X_valid is not None and y_valid is not None:
            validation_data = (X_valid, y_valid)
        else:
            validation_data = None

        history = self.lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=validation_data, callbacks=callbacks, verbose=2)

        return history

    def predict_lrcn(self, X_video_frames):
        return self.lrcn_model.predict(X_video_frames)
        # return self.cnn.predict(X_video_frames)

    def get_tf_model(self):
        return (self.cnn, self.lstm, self.lrcn_model)

    def set_tf_model(self, models):
        if models is None:
            self.cnn, self.lstm, self.lrcn_model = None, None, None
        else:
            self.cnn, self.lstm, self.lrcn_model = models

    def save(self, filename):
        self.lstm.save(filename, overwrite=True, save_format='h5')

    def load_model(self, filename):
        self.lstm = tf.keras.models.load_model(filename)
        self.cnn = self.create_cnn(self.params)
        self.lrcn_model = self.create_lrcn(self.params)

