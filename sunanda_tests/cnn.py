from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import logging
from keras_callbacks import ModelCheckpointer, Metrics, WeightRestorer, StopperOnGoal
from sklearn.utils import class_weight
import numpy as np


class CNN:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params
        self.cnn = Sequential()

        self.__add_layers_to_model()

        self.cnn.compile(optimizer='adam', loss='binary_crossentropy')

        # Following is also valid: need to explore more. But when using binary_crossentropy loss, set metrics=[categorical_accuracy]
        # self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        logger = logging.getLogger(__name__)
        self.cnn.summary(print_fn=logger.info)

    def __add_layers_to_model(self):
        layers_defs = self.params['layer_defs']

        is_first_dense = True

        # First Conv layer (need to specify input_shape here)
        layer = layers_defs[0]
        input_shape = (self.params['img_shape_x'], self.params['img_shape_y'], self.params['channels'])
        self.cnn.add(Conv2D(32, kernel_size=layer['kernel_size'], strides=layer['strides'],
                            activation=layer['activation'], input_shape=input_shape))

        # Other hidden layers
        for i in range(1, len(layers_defs)):
            layer = layers_defs[i]

            if layer['type'] == 'Conv':
                self.cnn.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'],
                                    strides=layer['strides'], activation=layer['activation']))
            elif layer['type'] == 'MaxPool':
                self.cnn.add(MaxPooling2D(layer['pool_size']))
            elif layer['type'] == 'Dense':
                if is_first_dense:
                    self.cnn.add(Flatten())
                    is_first_dense = False
                self.cnn.add(Dense(units=layer['units'], activation=layer['activation']))
            elif layer['type'] == 'Dropout':
                self.cnn.add(Dropout(layer['dropout_rate']))
            else:
                assert False    # Unknown layer type

        # # Output layer    # Defined in layer_defs
        # self.cnn.add(Dense(output_nodes, activation='softmax'))

    def set_layers_weights(self, all_layers_weights):
        dense_layers = [layer for layer in self.cnn.layers if type(layer) is Dense]
        dense_layers = dense_layers[:-1]    # Ignore the output layer

        assert len(dense_layers) == len(all_layers_weights)

        for layer, weights in zip(dense_layers, all_layers_weights):
            layer.set_weights(weights)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    # def fit(self, X, y):
    #     return

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, parent=None):
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        early_stop_patience = self.params['early_stop_patience']

        # --------------------------------
        # Create callbacks

        # tensorboard_cb = TensorBoard(log_dir=self.params['tensorboard_log_dir'], batch_size=batch_size)

        # interval = max(epochs // 10, 10)
        # checkpointer_cb = ModelCheckpointer(model_wrapper=self, checkpoint_epoch_interval=interval,
        #                                     save_directory=self.params['results_dir'], filename_prefix='ann')

        # metrics_cb = Metrics()

        # callbacks = [tensorboard_cb, checkpointer_cb, metrics_cb]
        # callbacks = [tensorboard_cb]
        callbacks = []

        # # Stop training when a goal metric value is reached
        # if 'goal_metric' in self.params and self.params['goal_metric'] > 0:
        #     assert X_test is not None
        #     assert y_test is not None
        #     if parent is None:
        #         model_wrapper = self
        #     else:
        #         model_wrapper = parent
        #     goal_stopper_cb = StopperOnGoal(model_wrapper, X_test, y_test, self.params['goal_metric'], "accuracy")
        #     callbacks.append(goal_stopper_cb)
        #
        # if X_valid is not None:
        #     weight_restorer_cb = WeightRestorer(epochs)
        #     callbacks.append(weight_restorer_cb)

        # Early stopping is enabled
        if X_valid is not None and early_stop_patience > 0 and early_stop_patience < epochs:
            early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                              verbose=2, mode='auto', restore_best_weights=False)
            callbacks.append(early_stopping_cb)

        # --------------------------------
        # Class weights to according to imbalance

        # class_weights = None
        # if self.params['class_weights'] != 0:
        #     logging.info("Computing class weights to according to imbalance")
        #     y_ints = [y.argmax() for y in y_train]
        #     class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
        #     logging.info("Class weights below.\n{}".format(class_weights))

        class_weights = None    # Disabled

        # --------------------------------
        # Fit

        if X_valid is not None and y_valid is not None:
            validation_data = (X_valid, y_valid)
        else:
            validation_data = None

        history = self.cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=validation_data, callbacks=callbacks,
                               class_weight=class_weights, verbose=1)

        # history.history['val_f1'] = metrics_cb.val_f1s

        # --------------------------------
        # Evaluate loss of final (best-weights-restored) model
        # These values can be different from the values seen during training (due to batch norm and dropout)

        train_loss = self.cnn.evaluate(X_train, y_train, verbose=0)
        val_loss = -1
        if X_valid is not None and y_valid is not None:
            val_loss = self.cnn.evaluate(X_valid, y_valid, verbose=0)
        logging.info('Last epoch loss evaluation: train_loss = {:.6f}, val_loss = {:.6f}'.format(train_loss, val_loss))

        return history

    def predict(self, X):
        # return self.predict_classes(X)
        return self.cnn.predict(X)

    def predict_classes(self, X):
        return self.cnn.predict_classes(X)

    def get_last_hidden_layer_activations(self, X):
        # Temp model from input to last layer
        last_layer_model = Model(inputs=self.cnn.input, outputs=self.cnn.layers[-2].output) # -2 is the last hidden layer

        last_layer_output = last_layer_model.predict(X)
        return last_layer_output

    def save(self, filename, **kwargs):
        self.cnn.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
