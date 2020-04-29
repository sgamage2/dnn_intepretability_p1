import tensorflow.compat.v1 as tf1


def tf_print(tensor, prefix, print_tensor_vals=True):
    print('{}{}'.format(prefix, tensor))

    if print_tensor_vals:
        sess = tf1.Session()
        with sess.as_default():
            print(tensor.eval())

