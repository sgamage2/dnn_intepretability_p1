# import saliency
from keras import backend as K
import tensorflow as tf


def get_gradient_saliency_scores(model, X, output_layer_idx):
    """ Computes gradient saliency significance scores for a given set of examples using the given model

    Gradients of output (at the last layer before the softmax) w.r.t input is taken and squared

    :param model: trained Keras model
    :param X: set of examples to compute feature significance scores on. Has shape: (num_samples, num_features)
    :param output_layer_idx: Last layer before the softmax
    :return: gradient_saliency: gradient saliency score of every feature of every example in X (array of the same shape as X)
    """

    assert False    # Not implemented yet

    output_tensor = model.layers[output_layer_idx].output
    # input_tensor = model.input
    input_tensor = X[0]

    # model.compile(optimizer='adam', loss='binary_crossentropy')

    tf.summary.trace_on(graph=True, profiler=True)

    gradients = K.gradients(output_tensor, input_tensor)

    print(gradients)

    # import tensorflow.compat.v1 as tfc
    # sess = K.get_session()
    # gradients = K.gradients(output_tensor, input_tensor)
    # # sess = tfc.InteractiveSession()
    # sess.run(tfc.initialize_all_variables())
    # evaluated_gradients = sess.run(gradients, feed_dict={model.input: X[0]})
    # print(evaluated_gradients)


if __name__ == '__main__':
    assert False    # Not meant to be run as a standalone script
