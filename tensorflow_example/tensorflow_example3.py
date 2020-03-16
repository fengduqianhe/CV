# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/21 14:03
# version： Python 3.7.8
# @File : tensorflow_example3.py
# @Software: PyCharm
import tensorflow as tf
#创建一个tf常量 1*2
matrix1 = tf.constant([[3, 3]])
#创建一个常量  2*1
matrix2 = tf.constant([[2], [2]])

#创建一个op
product = tf.matmul(matrix1, matrix2)

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)


input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

#初始化
init = tf.initialize_all_variables()
# print(sess.run(product))
# sess.close()

with tf.Session() as sess:
    sess.run(init)
    with tf.device("/gpu:1"):
        # for _ in range(50):
        #     result = sess.run(update)
        #     print(result)
        result = sess.run([mul, intermed])
        print(result)

