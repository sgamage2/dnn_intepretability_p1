from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import logging
from sklearn.utils import class_weight
import numpy as np
import keras
from models.keras_callbacks import WeightRestorer

from keras.losses import binary_crossentropy
import keras.backend as K


def binary_crossentropy_flood_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred)
    b = 0.02
    flood_loss = K.abs(loss - b) + b
    return flood_loss


class ANN:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params
        self.ann = Sequential()

        input_nodes = exp_params['ann_input_nodes']
        output_nodes = exp_params['output_nodes']
        layer_nodes = exp_params['ann_layer_units']
        activations = exp_params['ann_layer_activations']
        dropouts = exp_params['ann_layer_dropout_rates']

        # First hidden layer (need to specify input layer nodes here)
        self.ann.add(Dense(layer_nodes[0], input_dim=input_nodes, activation=activations[0],
                           kernel_initializer='he_uniform', bias_initializer='zeros'))
        # self.ann.add(BatchNormalization())
        self.ann.add(Dropout(dropouts[0]))

        # Other hidden layers
        for i in range(1, len(layer_nodes)):
            self.ann.add(Dense(layer_nodes[i], activation=activations[i],
                               kernel_initializer='he_uniform', bias_initializer='zeros'))
            # self.ann.add(BatchNormalization())  # BN after Activation (contested, but current preference)
            self.ann.add(Dropout(dropouts[i]))

        if output_nodes == 1:
            output_activation = 'sigmoid'
        elif output_nodes > 1:
            output_activation = 'softmax'

        self.ann.add(Dense(output_nodes, activation=output_activation,
                           kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self.ann.compile(optimizer='adam', loss='binary_crossentropy')
        # self.ann.compile(optimizer='adam', loss=binary_crossentropy_flood_loss)

        # Following is also valid: need to explore more. But when using binary_crossentropy loss, set metrics=[categorical_accuracy]
        # self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        logger = logging.getLogger(__name__)
        self.ann.summary(print_fn=logger.info)

    def set_layers_weights(self, all_layers_weights):
        dense_layers = [layer for layer in self.ann.layers if type(layer) is Dense]
        dense_layers = dense_layers[:-1]    # Ignore the output layer

        assert len(dense_layers) == len(all_layers_weights)

        for layer, weights in zip(dense_layers, all_layers_weights):
            layer.set_weights(weights)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        epochs = self.params['ann_epochs']
        batch_size = self.params['ann_batch_size']
        early_stop_patience = self.params['ann_early_stop_patience']


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
        # Class weights to according to imbalance

        class_weights = None
        if self.params['ann_class_weights'] != 0:
            logging.info("Computing class weights to according to imbalance")
            # y_ints = [y.argmax() for y in y_train]
            y_ints = y_train
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
            logging.info("Class weights below.\n{}".format(class_weights))


        # --------------------------------
        # Fit

        if X_valid is not None and y_valid is not None:
            validation_data = (X_valid, y_valid)
        else:
            validation_data = None

        history = self.ann.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=validation_data, callbacks=callbacks,
                               class_weight=class_weights, verbose=2)

        # --------------------------------
        # Evaluate loss of final (best-weights-restored) model
        # These values can be different from the values seen during training (due to batch norm and dropout)


        train_loss = self.ann.evaluate(X_train, y_train, verbose=0)
        val_loss = -1
        if X_valid is not None and y_valid is not None:
            val_loss = self.ann.evaluate(X_valid, y_valid, verbose=0)
        logging.info('Last epoch loss evaluation: train_loss = {:.6f}, val_loss = {:.6f}'.format(train_loss, val_loss))

        return history

    def predict(self, X):
        # return self.predict_classes(X)
        return self.ann.predict(X)

    def predict_classes(self, X):
        return self.ann.predict_classes(X)

    def get_last_hidden_layer_activations(self, X):
        # Temp model from input to last layer
        last_layer_model = Model(inputs=self.ann.input, outputs=self.ann.layers[-2].output) # -2 is the last hidden layer

        last_layer_output = last_layer_model.predict(X)
        return last_layer_output

    def save(self, filename, **kwargs):
        self.ann.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
