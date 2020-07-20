# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 14:45:02 2020

@author: daisy
"""
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow.compat.v1 as tf
#import tensorflow as tf

import os
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope


import ops
import scopes


class MNIST_NN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=True, name='layer1')
            dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu, use_bias=True, name='layer2')
            logits = tf.layers.dense(inputs=dense2, units=10, activation=None, use_bias=True, name='layer3')
            prediction = tf.nn.softmax(logits)


        return [dense1, dense2, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=True)
            dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu, use_bias=True)
            dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu, use_bias=True)
            dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu, use_bias=True)
            logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=True)
            prediction = tf.nn.softmax(logits)

        return [dense1, dense2, dense3, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class InceptionV3_Network:

    def __init__(self, name):
        self.name = name


    def __call__(self, inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope='',
                 reuse=False):

        end_points = {}

        with tf.name_scope(scope, 'inception_v3', [inputs]):

            with tf.name_scope(scope, 'inception_v3', [inputs]):
                with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                                      is_training=is_training):
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                          stride=1, padding='VALID') as scope:
                    # 299 x 299 x 3
                    end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                                     scope='conv0', reuse=reuse)
                    # 149 x 149 x 32
                    end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                                     scope='conv1', reuse=reuse)
                    # 147 x 147 x 32
                    end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                                     padding='SAME', scope='conv2', reuse=reuse)
                    # 147 x 147 x 64
                    end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                                       stride=2, scope='pool1')
                    # 73 x 73 x 64
                    end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                                     scope='conv3', reuse=reuse)
                    # 73 x 73 x 80.
                    end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                                     scope='conv4', reuse=reuse)
                    # 71 x 71 x 192.
                    end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                                       stride=2, scope='pool2')
                    # 35 x 35 x 192.
                    net = end_points['pool2']
                # Inception blocks
                with arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                      stride=1, padding='SAME'):
                    # mixed: 35 x 35 x 256.
                    with tf.variable_scope('mixed_35x35x256a'):
                        with tf.variable_scope('branch1x1') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch1x1 = ops.conv2d(net, 64, [1, 1])
                        with tf.variable_scope('branch5x5') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                        with tf.variable_scope('branch3x3dbl') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        with tf.variable_scope('branch_pool') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
                        net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                        end_points['mixed_35x35x256a'] = net
                    # mixed_1: 35 x 35 x 288.
                    with tf.variable_scope('mixed_35x35x288a'):
                        with tf.variable_scope('branch1x1') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch1x1 = ops.conv2d(net, 64, [1, 1])
                        with tf.variable_scope('branch5x5') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                        with tf.variable_scope('branch3x3dbl') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        with tf.variable_scope('branch_pool') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                        net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                        end_points['mixed_35x35x288a'] = net
                    # mixed_2: 35 x 35 x 288.
                    with tf.variable_scope('mixed_35x35x288b'):
                        with tf.variable_scope('branch1x1') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch1x1 = ops.conv2d(net, 64, [1, 1])
                        with tf.variable_scope('branch5x5') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                        with tf.variable_scope('branch3x3dbl') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                        with tf.variable_scope('branch_pool') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                        net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
                        end_points['mixed_35x35x288b'] = net
                    # mixed_3: 17 x 17 x 768.
                    with tf.variable_scope('mixed_17x17x768a'):
                        with tf.variable_scope('branch3x3') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                        with tf.variable_scope('branch3x3dbl') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                                      stride=2, padding='VALID')
                        with tf.variable_scope('branch_pool') as scope:
                            if reuse:
                                scope.reuse_variables()
                            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                        net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
                        end_points['mixed_17x17x768a'] = net
                    # mixed4: 17 x 17 x 768.
                    with tf.variable_scope('mixed_17x17x768b') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 192, [1, 1])
                        with tf.variable_scope('branch7x7'):
                            branch7x7 = ops.conv2d(net, 128, [1, 1])
                            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                        with tf.variable_scope('branch7x7dbl'):
                            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                        end_points['mixed_17x17x768b'] = net
                    # mixed_5: 17 x 17 x 768.
                    with tf.variable_scope('mixed_17x17x768c') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 192, [1, 1])
                        with tf.variable_scope('branch7x7'):
                            branch7x7 = ops.conv2d(net, 160, [1, 1])
                            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                        with tf.variable_scope('branch7x7dbl'):
                            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                        end_points['mixed_17x17x768c'] = net
                    # mixed_6: 17 x 17 x 768.
                    with tf.variable_scope('mixed_17x17x768d') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 192, [1, 1])
                        with tf.variable_scope('branch7x7'):
                            branch7x7 = ops.conv2d(net, 160, [1, 1])
                            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                        with tf.variable_scope('branch7x7dbl'):
                            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                        end_points['mixed_17x17x768d'] = net
                    # mixed_7: 17 x 17 x 768.
                    with tf.variable_scope('mixed_17x17x768e') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 192, [1, 1])
                        with tf.variable_scope('branch7x7'):
                            branch7x7 = ops.conv2d(net, 192, [1, 1])
                            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                        with tf.variable_scope('branch7x7dbl'):
                            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
                        end_points['mixed_17x17x768e'] = net
                    # Auxiliary Head logits
                    aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                    with tf.variable_scope('aux_logits') as scope:
                        if reuse:
                            scope.reuse_variables()
                        aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                                  padding='VALID')
                        aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
                        # Shape of feature map before the final layer.
                        shape = aux_logits.get_shape()
                        aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                                padding='VALID')
                        aux_logits = ops.flatten(aux_logits)
                        aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                                            stddev=0.001, restore=restore_logits)
                        end_points['aux_logits'] = aux_logits
                    # mixed_8: 8 x 8 x 1280.
                    # Note that the scope below is not changed to not void previous
                    # checkpoints.
                    # (TODO) Fix the scope when appropriate.
                    with tf.variable_scope('mixed_17x17x1280a') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch3x3'):
                            branch3x3 = ops.conv2d(net, 192, [1, 1])
                            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                                   padding='VALID')
                        with tf.variable_scope('branch7x7x3'):
                            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                                     stride=2, padding='VALID')
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                        net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
                        end_points['mixed_17x17x1280a'] = net
                    # mixed_9: 8 x 8 x 2048.
                    with tf.variable_scope('mixed_8x8x2048a') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 320, [1, 1])
                        with tf.variable_scope('branch3x3'):
                            branch3x3 = ops.conv2d(net, 384, [1, 1])
                            branch3x3 = tf.concat(3, [ops.conv2d(branch3x3, 384, [1, 3]),
                                                      ops.conv2d(branch3x3, 384, [3, 1])])
                        with tf.variable_scope('branch3x3dbl'):
                            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                            branch3x3dbl = tf.concat(3, [ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                         ops.conv2d(branch3x3dbl, 384, [3, 1])])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                        end_points['mixed_8x8x2048a'] = net
                    # mixed_10: 8 x 8 x 2048.
                    with tf.variable_scope('mixed_8x8x2048b') as scope:
                        if reuse:
                            scope.reuse_variables()
                        with tf.variable_scope('branch1x1'):
                            branch1x1 = ops.conv2d(net, 320, [1, 1])
                        with tf.variable_scope('branch3x3'):
                            branch3x3 = ops.conv2d(net, 384, [1, 1])
                            branch3x3 = tf.concat(3, [ops.conv2d(branch3x3, 384, [1, 3]),
                                                      ops.conv2d(branch3x3, 384, [3, 1])])
                        with tf.variable_scope('branch3x3dbl'):
                            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                            branch3x3dbl = tf.concat(3, [ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                         ops.conv2d(branch3x3dbl, 384, [3, 1])])
                        with tf.variable_scope('branch_pool'):
                            branch_pool = ops.avg_pool(net, [3, 3])
                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                        net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
                        end_points['mixed_8x8x2048b'] = net
                    # Final pooling and prediction
                    with tf.variable_scope('logits'):
                        shape = net.get_shape()
                        net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
                        # 1 x 1 x 2048
                        net = ops.dropout(net, dropout_keep_prob, scope='dropout')
                        net = ops.flatten(net, scope='flatten')
                        # 2048
                        logits = ops.fc(net, num_classes, activation=None, scope='logits',
                                        restore=restore_logits, reuse=reuse)
                        # 1000
                        end_points['logits'] = logits
                        end_points['predictions'] = tf.nn.softmax(logits, name='predictions')

                # logits, endpoints["aux_logits"]
                return end_points, logits


    def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                                batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        """Yields the scope with the default parameters for inception_v3.
        Args:
          weight_decay: the weight decay for weights variables.
          stddev: standard deviation of the truncated guassian weight distribution.
          batch_norm_decay: decay for the moving average of batch_norm momentums.
          batch_norm_epsilon: small float added to variance to avoid dividing by zero.
        Yields:
          a arg_scope with the parameters needed for inception_v3.
        """
        # Set weight_decay for weights in Conv and FC layers.
        with arg_scope([ops.conv2d, ops.fc],
                              weight_decay=weight_decay):
            # Set stddev, activation and parameters for batch_norm.
            with arg_scope([ops.conv2d],
                                  stddev=stddev,
                                  activation=tf.nn.relu,
                                  batch_norm_params={
                                      'decay': batch_norm_decay,
                                      'epsilon': batch_norm_epsilon}) as arg_scope:
                yield arg_scope


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class LRP:

    def __init__(self, alpha, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, name):
        self.alpha = alpha
        self.activations = activations
        self.weights = weights
        self.biases = biases
        self.conv_ksize = conv_ksize
        self.pool_ksize = pool_ksize
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.name = name

    def __call__(self, logit):

        with tf.name_scope(self.name):
            Rs = []
            j = 0

            for i in range(len(self.activations) - 1):

                if i is 0:
                    Rs.append(self.activations[i][:,logit,None])
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j][:,logit,None], self.biases[j][logit,None], Rs[-1]))
                    j += 1

                    continue

                elif 'dense' in self.activations[i].name.lower():
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j], self.biases[j], Rs[-1]))
                    j += 1
                elif 'reshape' in self.activations[i].name.lower():
                    shape = self.activations[i + 1].get_shape().as_list()
                    shape[0] = -1
                    Rs.append(tf.reshape(Rs[-1], shape))
                elif 'conv' in self.activations[i].name.lower():
                    Rs.append(self.backprop_conv(self.activations[i + 1], self.weights[j], self.biases[j], Rs[-1], self.conv_strides))
                    j += 1
                elif 'pooling' in self.activations[i].name.lower():
                    if 'max' in self.activations[i].name.lower():
                        pooling_type = 'max'
                    else:
                        pooling_type = 'avg'
                    Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, pooling_type))
                else:
                    raise Error('Unknown operation.')

            return Rs[-1]

    def backprop_conv(self, activation, kernel, bias, relevance, strides, padding='SAME'):
        W_p = tf.maximum(0., kernel)
        b_p = tf.maximum(0., bias)
        z_p = nn_ops.conv2d(activation, W_p, strides, padding) + b_p
        s_p = relevance / z_p
        c_p = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s_p, strides, padding)

        W_n = tf.minimum(0., kernel)
        b_n = tf.minimum(0., bias)
        z_n = nn_ops.conv2d(activation, W_n, strides, padding) + b_n
        s_n = relevance / z_n
        c_n = nn_ops.conv2d_backprop_input(tf.shape(activation), W_n, s_n, strides, padding)

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)

    def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):

        if pooling_type.lower() is 'avg':
            z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
            return activation * c
        else:
            z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
            return activation * c

    def backprop_dense(self, activation, kernel, bias, relevance):
        W_p = tf.maximum(0., kernel)
        b_p = tf.maximum(0., bias)
        z_p = tf.matmul(activation, W_p) + b_p
        s_p = relevance / z_p
        c_p = tf.matmul(s_p, tf.transpose(W_p))

        W_n = tf.minimum(0., kernel)
        b_n = tf.minimum(0., bias)
        z_n = tf.matmul(activation, W_n) + b_n
        s_n = relevance / z_n
        c_n = tf.matmul(s_n, tf.transpose(W_n))

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)