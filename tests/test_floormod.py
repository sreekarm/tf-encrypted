import unittest

from typing import List

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestFloorMod(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_pad(self):

        with tfe.protocol.Pond() as prot:

            tf.reset_default_graph()

            x = np.array([2,5,6,7])
            x_input = prot.define_public_variable(x)

            y = np.array([5,2,6,3])
            y_input = prot.define_public_variable(y)

            out = prot.floormod(x_input, y_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tfe = sess.run(out)
                print(out_tfe)

            tf.reset_default_graph()
            
            out_tensorflow = run_floormod([4,1])

            np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)

def run_floormod(input_shape):
    x = tf.placeholder(tf.float32, shape=input_shape, name="input")
    y = tf.constant(np.array([5, 2, 6, 3]).reshape(4,1), dtype=tf.float32)

    out = tf.floormod(x, y)

    with tf.Session() as sess:
        output = sess.run(out, feed_dict={x: np.array([2, 5, 6, 7]).reshape(4,1)})

    return output

if __name__ == '__main__':
    unittest.main()
