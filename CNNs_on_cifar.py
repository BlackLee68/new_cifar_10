# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
from numpy import array 
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
import matplotlib.image as mpimg
# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline
## test the pickle file still looks good after unpickle the file
pickle_file = 'cifar_10_01_type.pickle'
with open(pickle_file, 'rb') as f:
    dic = pickle.load(f)
    train_dataset = dic['train_dataset']
    train_labels = dic['train_labels']
    valid_dataset = dic['valid_dataset']
    valid_labels = dic['valid_labels']
    test_dataset = dic['test_dataset']
    test_labels = dic['test_labels']
    print(train_dataset.shape)
    sample_train_data = train_dataset[2338, :, :, :]
    print(sample_train_data)
    imgplot = plt.imshow(sample_train_data)
    ## still looks not bad, better than last two pickle files
    ## hope the labels will be all right
    ## looks great!
## reformate the labels array
image_size = 32
num_labels = 10
num_channels = 3 # RGB channle

import numpy as np

def reformat(labels):
  #dataset = dataset.reshape(
    #(-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return labels
train_labels = reformat(train_labels)
valid_labels = reformat(valid_labels)
test_labels = reformat(test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print(test_labels[0, :])
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
batch_size = 16
patch_size = 5
depth = 8
num_hidden = 64
num_hidden_two = 32
graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth*num_channels], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth*num_channels]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*num_channels, depth*num_channels], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth*num_channels]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth*num_channels, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden_two], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_two]))
  layer5_weights = tf.Variable(tf.truncated_normal(
      [num_hidden_two, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    avg_pool_one = tf.nn.avg_pool(hidden,[1,2,2,1],[1,2,2,1], padding='SAME')
    conv = tf.nn.conv2d(avg_pool_one, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    avg_pool_two = tf.nn.avg_pool(hidden,[1,2,2,1],[1,2,2,1], padding='SAME')
    shape = avg_pool_two.get_shape().as_list()
    print(shape)
    reshape = tf.reshape(avg_pool_two, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
    return tf.matmul(hidden, layer5_weights) + layer5_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  # using the learning rate decay
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.08, global_step, 500, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
num_steps = 65001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
