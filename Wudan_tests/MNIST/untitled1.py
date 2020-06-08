# LRP FOR MNIST_DNN

# 1. import dependencies=========================================
import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np

from model import MNIST_NN, MNIST_DNN, LRP
from utils import pixel_range


# 2. import dataset ========================================
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

logdir = './LRP_wudan/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)
    
#  building graph=========================================

with tf.name_scope('Classifier'):

    # Initialize neural network
    DNN = MNIST_DNN('DNN')

    tf.disable_eager_execution()

    # Setup training process
    X = tf.placeholder(tf.float32, [None, 784], name='X')
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    
    activations, logits = DNN(X)
    
    tf.add_to_collection('LRP', X)
    
    for activation in activations:
        tf.add_to_collection('LRP', activation)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()

# train the network===========================================
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hyperparameters
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    avg_acc = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        avg_acc += a / total_batch
        
        file_writer.add_summary(summary_str, epoch * total_batch + i)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(avg_acc))
    
    saver.save(sess, ckptdir)

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

sess.close()

# restore subgraph
tf.reset_default_graph()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*kernel.*')
biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*bias.*')
activations = tf.get_collection('LRP')
X = activations[0]

# Attach Subgraph for Calculating Relevance Scores
conv_ksize = [1, 3, 3, 1]
pool_ksize = [1, 2, 2, 1]
conv_strides = [1, 1, 1, 1]
pool_strides = [1, 2, 2, 1]

weights.reverse()
biases.reverse()
activations.reverse()

# LRP-alpha1-beta0
lrp10 = LRP(1, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, 'LRP10')

# LRP-alpha2-beta1
lrp21 = LRP(2, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, 'LRP21')

Rs10 = [lrp10(i) for i in range(10)]
Rs21 = [lrp21(i) for i in range(10)]

# calculate relevance score
sample_imgs = []
for i in range(10):
    sample_imgs.append(images[np.argmax(labels, axis=1) == i][3])

imgs10 = []
imgs21 = []
for i in range(10):
    imgs10.append(sess.run(Rs10[i], feed_dict={X: sample_imgs[i][None,:]}))
    imgs21.append(sess.run(Rs21[i], feed_dict={X: sample_imgs[i][None,:]}))

sess.close()

#  LRP-alpha1-beta0
plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    vmin, vmax = pixel_range(imgs10[2 * i])
    plt.imshow(np.reshape(imgs10[2 * i], [28, 28]), vmin=-vmax, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    vmin, vmax = pixel_range(imgs10[2 * i + 1])
    plt.imshow(np.reshape(imgs10[2 * i + 1], [28, 28]), vmin=-vmax, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()

#  LRP-alpha2-beta1
plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    vmin, vmax = pixel_range(imgs21[2 * i])
    plt.imshow(np.reshape(imgs21[2 * i], [28, 28]), vmin=vmin, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    vmin, vmax = pixel_range(imgs21[2 * i + 1])
    plt.imshow(np.reshape(imgs21[2 * i + 1], [28, 28]), vmin=vmin, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()