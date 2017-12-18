# -*- coding: UTF-8 -*-
#@test {"output": "ignore"}
import tensorflow as tf
import numpy as np

with tf.Session():
    input_features = tf.constant(np.reshape([1, 0, 0, 1], (1, 4)).astype(np.float32))
    weights = tf.constant(np.random.randn(4, 2).astype(np.float32))
    output = tf.matmul(input_features, weights)
    print("Input:")
    print(input_features.eval())
    print("Weights:")
    print(weights.eval())
    print("Output:")
    print(output.eval())
    print ("@@@@@@@@@@@")
    print (np.random.randn(4, 2))