# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

with tf.Session():
    input1 = tf.constant(1.0, shape=[2, 3])
    input2 = tf.constant(np.reshape(np.arange(1.0, 7.0, dtype=np.float32), (2, 3)))
    output = tf.add(input1, input2)
    ara = np.arange(1.0, 7.0, dtype=np.float32)
    reshape = np.reshape(ara, (2, 3))
    print (np.reshape([1, 0, 2, 1], (2, 2)))
   # print(output.eval())