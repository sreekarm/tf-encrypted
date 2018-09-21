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
IN_DIM = 28
KERNEL = 5
STRIDE = 1
IN_CHANNELS = 1
HIDDEN_C1 = 6
HIDDEN_C2 = 16
HIDDEN_FC1 = 256
HIDDEN_FC2 = 120
HIDDEN_FC3 = 84
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


  def weight_variable(shape, gain):
      """weight_variable generates a weight variable of a given shape."""
      if len(shape) == 2:
          fan_in, fan_out = shape
      elif len(shape) == 4:
          h, w, c_in, c_out = shape
          fan_in = h * w * c_in
          fan_out = h * w * c_out
      r = gain * math.sqrt(6 / (fan_in + fan_out))
      initial = tf.random_uniform(shape, minval=-r, maxval=r)
      return tf.Variable(initial)


  def bias_variable(shape):
      """bias_variable generates a bias variable of a given shape."""
      initial = tf.constant(0., shape=shape)
      return tf.Variable(initial)


  conv2d = lambda x, w, s: tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='VALID')
  pooling = lambda x: tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')


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

  # model parameters and initial values
  Wconv1 = weight_variable([KERNEL,
                            KERNEL,
                            IN_CHANNELS,
                            HIDDEN_C1], 1.)
  bconv1 = bias_variable([1, 1, HIDDEN_C1])
  Wconv2 = weight_variable([KERNEL,
                            KERNEL,
                            HIDDEN_C1,
                            HIDDEN_C2], 1.)
  bconv2 = bias_variable([1, 1, HIDDEN_C2])
  Wfc1 = weight_variable([HIDDEN_FC1, HIDDEN_FC2], 1.)
  bfc1 = bias_variable([HIDDEN_FC2])
  Wfc2 = weight_variable([HIDDEN_FC2, HIDDEN_FC3], 1.)
  bfc2 = bias_variable([HIDDEN_FC3])
  Wfc3 = weight_variable([HIDDEN_FC3, OUT_N], 1.)
  bfc3 = bias_variable([OUT_N])
  params = [Wconv1, bconv1, Wconv2, bconv2, Wfc1, bfc1, Wfc2, bfc2, Wfc3, bfc3]

  # model construction
  x_image = tf.reshape(x, [-1, IN_DIM, IN_DIM, 1])
  layer1 = pooling(tf.nn.relu(conv2d(x_image, Wconv1, STRIDE) + bconv1))
  layer2 = pooling(tf.nn.relu(conv2d(layer1, Wconv2, STRIDE) + bconv2))
  layer2 = tf.reshape(layer2, [-1, HIDDEN_FC1])
  layer3 = tf.nn.relu(tf.matmul(layer2, Wfc1) + bfc1)
  layer4 = tf.nn.relu(tf.matmul(layer3, Wfc2) + bfc2)
  y = tf.matmul(layer4, Wfc3) + bfc3


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
      xs, ys = mnist.train.next_batch(BATCH_SIZE)
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
  np.save('../examples/test_data/network_c_tf_output_100.npy', pred)

  current_dir = os.getcwd()
  pb_filename = '/test_data/network_c.pb'
  export_to_pb(sess, y, current_dir + pb_filename)
  np_filename = '/test_data/mnist_input_network_c.npy'
  np.save(current_dir + np_filename, mnist.test.images[0].reshape(1,784))
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
