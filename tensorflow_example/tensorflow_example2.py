# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/21 13:52
# version： Python 3.7.8
# @File : tensorflow_example2.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/21 13:18
# version： Python 3.7.8
# @File : tensorflow_example1.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np

#  生成 2*100的 数据
x_data = np.float32(np.random.rand(2, 100))
print(x_data.shape)
# y = wx + b
y_data = np.dot([0.100, 0.200], x_data) + 0.300
print(y_data)

#设定初始参数值 b (1)  w  (1*2)
x = tf.placeholder(tf.float32, [2, 100], name="x")
y = tf.placeholder(tf.float32, [1, 100], name="y")
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y_output = tf.matmul(w, x) + b
#定义优化目标函数
loss = tf.reduce_mean(tf.square(y_output-y))
#构建优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化变量
init = tf.initialize_all_variables()
#启动图
sess = tf.Session()
#会话运行
sess.run(init)

for step in range(0,201):
    sess.run(train, feed_dict={x: x_data, y: y_data.reshape(1, -1)})
    if step % 20 == 0:
        print("第%s迭代，w为%s,b为%s" % (step, sess.run(w), sess.run(b)))


