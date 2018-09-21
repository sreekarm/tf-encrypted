# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import numpy as np

import math

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

BATCH_SIZE = 32
ITERATIONS = 60000 // BATCH_SIZE
EPOCHS = 15
IN_N = 28 * 28
HIDDEN_N = 128
OUT_N = 10

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  #with tf.name_scope('input_reshape'):
  #  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  # tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  # def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  #   """Reusable code for making a simple neural net layer.
  #
  #   It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  #   It also sets up name scoping so that the resultant graph is easy to read,
  #   and adds a number of summary ops.
  #   """
  #   # Adding a name scope ensures logical grouping of the layers in the graph.
  #   with tf.name_scope(layer_name):
  #     # This Variable will hold the state of the weights for the layer
  #     with tf.name_scope('weights'):
  #       weights = weight_variable([input_dim, output_dim])
  #       variable_summaries(weights)
  #     with tf.name_scope('biases'):
  #       biases = bias_variable([output_dim])
  #       variable_summaries(biases)
  #     with tf.name_scope('Wx_plus_b'):
  #       preactivate = tf.matmul(input_tensor, weights) + biases
  #       tf.summary.histogram('pre_activations', preactivate)
  #     activations = act(preactivate, name='activation')
  #     tf.summary.histogram('activations', activations)
  #     return activations
  #
  #
  #
  # hidden1 = nn_layer(x, 784, 500, 'layer1')
  #
  # with tf.name_scope('dropout'):
  #   keep_prob = tf.placeholder(tf.float32)
  #   tf.summary.scalar('dropout_keep_probability', keep_prob)
  #   dropped = tf.nn.dropout(hidden1, keep_prob)
  #
  # # Do not apply softmax activation yet, see below.
  # y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  j = IN_N
  k = HIDDEN_N
  m = OUT_N
  r_in = math.sqrt(12 / (j + k))
  r_hid = math.sqrt(12 / (2 * k))
  r_out = math.sqrt(12 / (k + m))

  # model parameters and initial values
  with tf.name_scope('weights_0'):
      w0 = tf.Variable(tf.random_uniform([j, k], minval=-r_in, maxval=r_in))
      variable_summaries(w0)
  with tf.name_scope('biases_0'):
      b0 = tf.Variable(tf.zeros([k]))
      variable_summaries(b0)
  with tf.name_scope('weights_1'):
      w1 = tf.Variable(tf.random_uniform([k, k], minval=-r_hid, maxval=r_hid))
      variable_summaries(w1)
  with tf.name_scope('biases_1'):
      b1 = tf.Variable(tf.zeros([k]))
      variable_summaries(b1)
  with tf.name_scope('weights_2'):
      w2 = tf.Variable(tf.random_uniform([k, m], minval=-r_out, maxval=r_out))
      variable_summaries(w2)
  with tf.name_scope('biases_1'):
      b2 = tf.Variable(tf.zeros([m]))
      variable_summaries(b2)
  params = [w0, b0, w1, b1, w2, b2]

  # model construction
  layer0 = x
  with tf.name_scope('W0x_plus_b0_bf_relu'):
      layer1_bf_relu = tf.matmul(layer0, w0) + b0
      tf.summary.histogram('layer1_bf_relu', layer1_bf_relu)

  with tf.name_scope('W0x_plus_b0_aft_relu'):
      layer1_aft_relu = tf.nn.relu(layer1_bf_relu)
      tf.summary.histogram('layer1_aft_relu', layer1_aft_relu)

  with tf.name_scope('W1x_plus_b1_bf_relu'):
      layer2_bf_relu = tf.matmul(layer1_aft_relu, w1) + b1
      tf.summary.histogram('layer2_bf_relu', layer2_bf_relu)

  with tf.name_scope('W1x_plus_b1_aft_relu'):
      layer2_aft_relu = tf.nn.relu(layer2_bf_relu)
      tf.summary.histogram('layer2_aft_relu', layer2_aft_relu)

  with tf.name_scope('W2x_plus_b2'):
      layer3 = tf.matmul(layer2_aft_relu, w2) + b2
      tf.summary.histogram('layer3', layer3)

  y = layer3

  # layer0 = x
  # layer1 = tf.nn.relu(tf.matmul(layer0, w0) + b0)
  # layer2 = tf.nn.relu(tf.matmul(layer1, w1) + b1)
  # layer3 = tf.matmul(layer2, w2) + b2
  # y = layer3

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          labels=y_, logits=y)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  print(FLAGS.log_dir )
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      #k = FLAGS.dropout
    else:
          xs, ys = mnist.test.images, mnist.test.labels
      #k = 1.0
    return {x: xs, y_: ys}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

  x_100, y_100 = mnist.test.images[:100], mnist.test.labels[:100]
  summary, acc, pred = sess.run([merged, accuracy, y], feed_dict={x: x_100, y_: y_100})
  print("Accuracy on batch_100", acc)
  np.save('../examples/test_data/network_a_tf_output_100.npy', pred)

  current_dir = os.getcwd()
  pb_filename = '/test_data/network_a.pb'
  export_to_pb(sess, y, current_dir + pb_filename)
  np_filename = '/test_data/mnist_input_network_a.npy'
  np.save(current_dir + np_filename, mnist.test.images[1].reshape(1,784))
  train_writer.close()
  test_writer.close()

from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

def export_to_pb(sess, x, filename):
    pred_names = ['output']
    tf.identity(x, name=pred_names[0])

    graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_names)

    graph = graph_util.remove_training_nodes(graph)
    path = graph_io.write_graph(graph, ".", filename, as_text=False)
    print('saved the frozen graph (ready for inference) at: ', filename)

    return path

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
