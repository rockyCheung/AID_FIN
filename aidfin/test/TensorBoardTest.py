#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
with tf.Session() as sess:
  merged_summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('/Users/zhangpenghong/Documents/logs', sess.graph)
  total_step = 0
  while training:
    total_step += 1
    sess.run(training_op)
    if total_step % 100 == 0:
      summary_str = sess.run(merged_summary_op)
      summary_writer.add_summary(summary_str, total_step)