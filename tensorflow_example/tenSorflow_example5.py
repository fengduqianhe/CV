# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/22 17:57
# version： Python 3.7.8
# @File : tenSorflow_example5.py
# @Software: PyCharm
# softmax 实现


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/minst/", one_hot=True)
#28*28 = 784  n * 784
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    print("第%s次迭代,交叉熵为%s" % (i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})))

# 模型评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))