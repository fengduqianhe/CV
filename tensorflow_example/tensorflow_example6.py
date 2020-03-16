# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/23 12:41
# version： Python 3.7.8
# @File : tensorflow_example6.py
# @Software: PyCharm
#转置卷积的验证例子


import tensorflow as tf

x = tf.reshape(tf.constant([[1,2],
                            [4,5]],dtype=tf.float32), [1, 2, 2, 1])
kernel = tf.reshape(tf.constant([[1,2,3],
                                 [4,5,6],
                                 [7,8,9]],dtype=tf.float32), [3, 3, 1, 1])
transpose_conv = tf.nn.conv2d_transpose(x, kernel, output_shape=[1, 4, 4, 1], strides=[1,1,1,1], padding='VALID')
sess = tf.Session()
print(sess.run(x))
print(sess.run(kernel))
print(sess.run(transpose_conv))


x2 = tf.reshape(tf.constant([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 2, 0, 0],
                             [0, 0, 4, 5, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]],dtype=tf.float32), [1, 6, 6, 1])
kernel2  = tf.reshape(tf.constant([[9,8,7],
                                   [6,5,4],
                                   [3,2,1]],dtype=tf.float32), [3, 3, 1, 1])
conv = tf.nn.conv2d(x2,kernel2,strides=[1,1,1,1],padding='VALID')

print(sess.run(x2))
print(sess.run(kernel2))
print(sess.run(conv))
