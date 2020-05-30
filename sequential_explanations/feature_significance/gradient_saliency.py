# import saliency
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K


def get_gradient_saliency_scores(model, X, output_layer_idx):
    """ Computes gradient saliency significance scores for a given set of examples using the given model

    Gradients of output (at the last layer before the softmax) w.r.t input is taken and squared

    :param model: trained Keras model
    :param X: set of examples to compute feature significance scores on. Has shape: (num_samples, num_features)
    :param output_layer_idx: Last layer before the softmax
    :return: gradient_saliency: gradient saliency score of every feature of every example in X (array of the same shape as X)
    """

    temp_output_layer = model.layers[output_layer_idx]
    temp_model = tf.keras.models.Model(inputs=model.input, outputs=temp_output_layer.output)

    X_ts = tf.convert_to_tensor(X)

    with tf.GradientTape() as tape:
        tape.watch(X_ts)
        pred_y = temp_model(X_ts)

    grad = tape.gradient(pred_y, X_ts)
    grad_inputs = grad.numpy()

    return grad_inputs


if __name__ == '__main__':
    assert False    # Not meant to be run as a standalone script
